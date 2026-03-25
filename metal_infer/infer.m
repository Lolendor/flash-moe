/*
 * infer.m — Qwen3.5 MoE inference engine using Metal
 *
 * Full forward pass: embedding -> N transformer layers -> norm -> lm_head -> sample
 * Model architecture loaded at runtime from HuggingFace config.json (--model flag).
 * Non-expert weights loaded from model_weights.bin (mmap'd at startup)
 * Expert weights loaded from packed_experts/ per layer per token (pread)
 *
 * Supported: Qwen3.5-35B-A3B, Qwen3.5-397B-A17B, and compatible MoE variants.
 * Architecture auto-detected from config.json:
 *   - N layers: mix of linear attention (GatedDeltaNet) + full attention
 *   - Configurable hidden_size, head_dim, num_attention_heads, num_kv_heads
 *   - Variable experts/layer and active experts (K)
 *   - Shared expert per layer (always active)
 *   - Linear attention: conv1d + gated delta recurrence
 *   - Full attention: standard QKV + scaled dot product + RoPE
 *
 * Command buffer optimization (fused_layer_forward):
 *   Per-layer Metal command buffer structure:
 *     CMD1: attention input projections (3-4 dispatches, 1 commit)
 *     CPU:  attention compute (RoPE/softmax/delta-net)
 *     CMD2: o_proj + residual_add + rms_norm + routing + shared gate/up (8 encoders, 1 commit)
 *           GPU handles residual connection and post-attn norm internally,
 *           eliminating the CPU round-trip that previously split this into 2 cmd buffers.
 *     CPU:  softmax + top-K + pread all K experts (4 pthreads parallel)
 *     CMD3: all K expert forwards + shared SwiGLU + shared down
 *           + GPU-side combine + residual_add + rms_norm -> buf_input (DEFERRED commit)
 *           Batched encoding: 4 encoders for K experts + 2 shared + 3 combine = 9 total
 *   Total: 3 cmd buffers per layer. CMD3 is submitted async (commit without wait).
 *   GPU-side combine in CMD3: for non-last layers, CMD3 also computes:
 *     moe_combine_residual (weighted sum + residual + shared gate -> hidden)
 *     rms_norm (hidden -> buf_input using NEXT layer's input_norm weights)
 *   This allows the next layer's CMD1 to submit immediately without waiting
 *   for CMD3 completion — the GPU queue serializes CMD3(N-1) then CMD1(N).
 *   Saves ~0.83ms/layer deferred_wait + CPU combine + input_norm overhead.
 *   Multi-expert buffers (MAX_K=16 independent slots) allow all K expert
 *   forwards to be encoded into a single command buffer.
 *   Batched encoding: 2 encoders per expert (gate+up fused, SwiGLU+down fused)
 *   + 2 for shared expert = K*2 + 2 total encoders in CMD3.
 *   Double-buffered expert data (buf_multi_expert_data / data_B) for future
 *   async pread overlap with GPU compute.
 *
 * Build:  clang -O2 -Wall -fobjc-arc -framework Metal -framework Foundation -lpthread infer.m -o infer
 * Run:    ./infer --prompt "Explain relativity" --tokens 50
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <math.h>
#include <getopt.h>
#include <pthread.h>
#include <errno.h>
#include <dispatch/dispatch.h>
#include <Accelerate/Accelerate.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/wait.h>
#include <compression.h>
#include <stdatomic.h>

// ============================================================================
// Runtime model configuration (populated from HuggingFace config.json)
// ============================================================================

#include "infer_api.h"

// ---- Compatibility macros for incoming Q3/batched-prefill code ----
// The incoming branch (9d1d602) used cfg.* and compile-time macros; HEAD uses cfg.* and runtime config.
// These bridge the gap so incoming code compiles against HEAD's ModelConfig.
#define MAX_LAYERS          64
#define MAX_EXPERTS         512
#define MAX_HIDDEN_DIM      4096
#define EXPERT_SIZE         ((int)cfg.expert_size_4bit)
#define EXPERT_SIZE_2BIT    ((int)cfg.expert_size_2bit)
#define GATE_W_OFF_2        ((int)cfg.gate_w_off_2)
#define GATE_S_OFF_2        ((int)cfg.gate_s_off_2)
#define GATE_B_OFF_2        ((int)cfg.gate_b_off_2)
#define UP_W_OFF_2          ((int)cfg.up_w_off_2)
#define UP_S_OFF_2          ((int)cfg.up_s_off_2)
#define UP_B_OFF_2          ((int)cfg.up_b_off_2)
#define DOWN_W_OFF_2        ((int)cfg.down_w_off_2)
#define DOWN_S_OFF_2        ((int)cfg.down_s_off_2)
#define DOWN_B_OFF_2        ((int)cfg.down_b_off_2)
// Q3 expert sizes (from incoming branch's GGUF Q3 support)
#define IQ3_XXS_EXPERT_PROJ_SIZE 1605632
#define EXPERT_SIZE_Q3_HYBRID    5439488
#define GATE_W_OFF_Q3  0
#define UP_W_OFF_Q3    (GATE_W_OFF_Q3 + IQ3_XXS_EXPERT_PROJ_SIZE)
#define DOWN_W_OFF_Q3  (UP_W_OFF_Q3   + IQ3_XXS_EXPERT_PROJ_SIZE)
#define DOWN_S_OFF_Q3  DOWN_W_OFF_Q3
#define DOWN_B_OFF_Q3  DOWN_W_OFF_Q3
#define IQ4_XS_EXPERT_PROJ_SIZE              2228224
#define Q5_K_EXPERT_PROJ_SIZE                2883584
#define EXPERT_SIZE_Q3_OUTLIER               7340032
#define GATE_W_OFF_Q3_OUTLIER  0
#define UP_W_OFF_Q3_OUTLIER    (GATE_W_OFF_Q3_OUTLIER + IQ4_XS_EXPERT_PROJ_SIZE)
#define DOWN_W_OFF_Q3_OUTLIER  (UP_W_OFF_Q3_OUTLIER   + IQ4_XS_EXPERT_PROJ_SIZE)
#define DOWN_S_OFF_Q3_OUTLIER  DOWN_W_OFF_Q3_OUTLIER
#define DOWN_B_OFF_Q3_OUTLIER  DOWN_W_OFF_Q3_OUTLIER
#define Q3_OUTLIER_LAYER       27
#define NUM_LAYERS          (cfg.num_layers)
#define NUM_EXPERTS         (cfg.num_experts)
#define HIDDEN_DIM          (cfg.hidden_dim)
#define RMS_NORM_EPS        (cfg.rms_norm_eps)
#define NUM_ATTN_HEADS      (cfg.num_attn_heads)
#define NUM_KV_HEADS        (cfg.num_kv_heads)
#define HEAD_DIM            (cfg.head_dim)
#define GROUP_SIZE          (cfg.group_size)
#define ROTARY_DIM          (cfg.rotary_dim)
#define SHARED_INTERMEDIATE (cfg.shared_intermediate)
#define MOE_INTERMEDIATE    (cfg.moe_intermediate)
#define VOCAB_SIZE          (cfg.vocab_size)
#define ROPE_THETA          (cfg.rope_theta)
#define LINEAR_CONV_DIM     (cfg.linear_conv_dim)
#define LINEAR_TOTAL_VALUE  (cfg.linear_total_value)
#define LINEAR_TOTAL_KEY    (cfg.linear_total_key)
#define LINEAR_NUM_V_HEADS  (cfg.linear_num_v_heads)
#define LINEAR_NUM_K_HEADS  (cfg.linear_num_k_heads)
#define LINEAR_KEY_DIM      (cfg.linear_key_dim)
#define LINEAR_VALUE_DIM    (cfg.linear_value_dim)

ModelConfig cfg;

// ---- Tiered expert quantization manifest ----
// Per-expert metadata: offset in layer file, size, and quant bits (2 or 4)
typedef struct {
    size_t offset;   // byte offset in layer_XX.bin
    size_t size;     // bytes to read (expert_size_4bit or expert_size_2bit)
    int bits;        // 2 or 4
} TieredExpertInfo;

// Global tiered manifest: NULL if not using tiered mode
static TieredExpertInfo *g_tiered_manifest = NULL;  // [num_layers * num_experts]
static int g_use_tiered = 0;

// Access helper
#define TIERED(l, e) g_tiered_manifest[(l) * cfg.num_experts + (e)]

static void compute_expert_offsets(ModelConfig *c) {
    int mid = c->moe_intermediate;
    int hid = c->hidden_dim;
    int gs = c->group_size;

    for (int b = 4; b >= 2; b -= 2) {
        int vals_per_u32 = 32 / b;
        // gate_proj [mid, hid]
        size_t gw = (size_t)mid * ((hid + vals_per_u32 - 1) / vals_per_u32) * 4;
        size_t gs_sz = (size_t)mid * ((hid + gs - 1) / gs) * 2;
        size_t gb = gs_sz;
        // up_proj [mid, hid] — same shape
        size_t uw = gw, us = gs_sz, ub = gb;
        // down_proj [hid, mid]
        size_t dw = (size_t)hid * ((mid + vals_per_u32 - 1) / vals_per_u32) * 4;
        size_t ds = (size_t)hid * ((mid + gs - 1) / gs) * 2;
        size_t db = ds;

        size_t off = 0;
        if (b == 4) {
            c->gate_w_off_4 = off; off += gw;
            c->gate_s_off_4 = off; off += gs_sz;
            c->gate_b_off_4 = off; off += gb;
            c->up_w_off_4   = off; off += uw;
            c->up_s_off_4   = off; off += us;
            c->up_b_off_4   = off; off += ub;
            c->down_w_off_4 = off; off += dw;
            c->down_s_off_4 = off; off += ds;
            c->down_b_off_4 = off; off += db;
            c->expert_size_4bit = off;
        } else {
            c->gate_w_off_2 = off; off += gw;
            c->gate_s_off_2 = off; off += gs_sz;
            c->gate_b_off_2 = off; off += gb;
            c->up_w_off_2   = off; off += uw;
            c->up_s_off_2   = off; off += us;
            c->up_b_off_2   = off; off += ub;
            c->down_w_off_2 = off; off += dw;
            c->down_s_off_2 = off; off += ds;
            c->down_b_off_2 = off; off += db;
            c->expert_size_2bit = off;
        }
    }
}

void load_model_config(const char *model_dir) {
    memset(&cfg, 0, sizeof(cfg));
    cfg.think_start_token = -1;
    cfg.think_end_token = -1;
    cfg.gpu_kv_seq = 8192;

    if (!model_dir || !model_dir[0]) {
        fprintf(stderr, "FATAL: --model path required\n");
        exit(1);
    }

    // Resolve HF snapshot directory
    NSString *base = [NSString stringWithUTF8String:model_dir];
    NSString *configPath = [base stringByAppendingPathComponent:@"config.json"];
    NSFileManager *fm = [NSFileManager defaultManager];

    if (![fm fileExistsAtPath:configPath]) {
        NSString *snapDir = [base stringByAppendingPathComponent:@"snapshots"];
        if ([fm fileExistsAtPath:snapDir]) {
            NSArray *snaps = [[fm contentsOfDirectoryAtPath:snapDir error:nil]
                              sortedArrayUsingSelector:@selector(compare:)];
            for (NSString *snap in snaps) {
                NSString *candidate = [[snapDir stringByAppendingPathComponent:snap]
                                        stringByAppendingPathComponent:@"config.json"];
                if ([fm fileExistsAtPath:candidate]) {
                    base = [snapDir stringByAppendingPathComponent:snap];
                    configPath = candidate;
                    break;
                }
            }
        }
    }

    if (![fm fileExistsAtPath:configPath]) {
        fprintf(stderr, "FATAL: config.json not found in %s\n", model_dir);
        exit(1);
    }

    strlcpy(cfg.model_path, [base UTF8String], sizeof(cfg.model_path));

    // Parse config.json
    NSData *data = [NSData dataWithContentsOfFile:configPath];
    NSError *jsonErr = nil;
    NSDictionary *root = [NSJSONSerialization JSONObjectWithData:data options:0 error:&jsonErr];
    if (!root) {
        fprintf(stderr, "FATAL: failed to parse config.json: %s\n", [[jsonErr localizedDescription] UTF8String]);
        exit(1);
    }
    // ---- Detect model type and resolve config dict ----
    // Qwen3.5 wraps model params in "text_config", MiniMax is flat.
    NSDictionary *tc = root[@"text_config"];
    NSString *model_type_str = root[@"model_type"];
    if (!model_type_str && tc) model_type_str = tc[@"model_type"];

    if ([model_type_str isEqualToString:@"minimax_m2"]) {
        cfg.model_type = 1;
        if (!tc) tc = root;  // MiniMax: flat config
        strlcpy(cfg.moe_prefix, "block_sparse_moe", sizeof(cfg.moe_prefix));
        cfg.has_attn_gate = 0;
        cfg.scoring_func = 1;  // sigmoid
        cfg.qk_norm_per_layer = 1;
        fprintf(stderr, "[config] Detected model: minimax_m2\n");
    } else {
        // Default: Qwen3.5 MoE
        cfg.model_type = 0;
        if (!tc) { fprintf(stderr, "FATAL: config.json missing text_config\n"); exit(1); }
        strlcpy(cfg.moe_prefix, "mlp", sizeof(cfg.moe_prefix));
        cfg.has_attn_gate = 1;
        cfg.scoring_func = 0;  // softmax
        cfg.qk_norm_per_layer = 0;
        fprintf(stderr, "[config] Detected model: qwen3_5_moe\n");
    }

    // ---- Core architecture (same field names for both) ----
    cfg.hidden_dim       = [tc[@"hidden_size"] intValue];
    cfg.num_layers       = [tc[@"num_hidden_layers"] intValue];
    cfg.num_attn_heads   = [tc[@"num_attention_heads"] intValue];
    cfg.num_kv_heads     = [tc[@"num_key_value_heads"] intValue];
    cfg.head_dim         = tc[@"head_dim"] ? [tc[@"head_dim"] intValue] : (cfg.hidden_dim / cfg.num_attn_heads);
    cfg.vocab_size       = [tc[@"vocab_size"] intValue];
    cfg.rms_norm_eps     = [tc[@"rms_norm_eps"] floatValue];
    cfg.max_seq_len      = [tc[@"max_position_embeddings"] intValue];

    // ---- MoE dimensions (field names differ per model) ----
    if (cfg.model_type == 1) {
        // MiniMax: num_local_experts, intermediate_size, shared_intermediate_size
        cfg.num_experts      = [tc[@"num_local_experts"] intValue];
        cfg.num_experts_per_tok = [tc[@"num_experts_per_tok"] intValue];
        cfg.moe_intermediate = [tc[@"intermediate_size"] intValue];
        cfg.shared_intermediate = [tc[@"shared_intermediate_size"] intValue];
    } else {
        // Qwen: num_experts, moe_intermediate_size, shared_expert_intermediate_size
        cfg.num_experts      = [tc[@"num_experts"] intValue];
        cfg.num_experts_per_tok = [tc[@"num_experts_per_tok"] intValue];
        cfg.moe_intermediate = [tc[@"moe_intermediate_size"] intValue];
        cfg.shared_intermediate = [tc[@"shared_expert_intermediate_size"] intValue];
    }

    // ---- Linear attention (GatedDeltaNet) — Qwen only, zero for MiniMax ----
    cfg.linear_num_v_heads = [tc[@"linear_num_value_heads"] intValue];
    cfg.linear_num_k_heads = [tc[@"linear_num_key_heads"] intValue];
    cfg.linear_key_dim   = tc[@"linear_key_head_dim"] ? [tc[@"linear_key_head_dim"] intValue] : 128;
    cfg.linear_value_dim = tc[@"linear_value_head_dim"] ? [tc[@"linear_value_head_dim"] intValue] : 128;
    cfg.conv_kernel_size = tc[@"linear_conv_kernel_dim"] ? [tc[@"linear_conv_kernel_dim"] intValue] : 4;

    // ---- Quantization ----
    NSDictionary *qc = root[@"quantization_config"] ?: root[@"quantization"];
    if (qc) {
        cfg.group_size = [qc[@"group_size"] intValue];
        cfg.bits       = [qc[@"bits"] intValue];
    } else {
        cfg.group_size = 64;
        cfg.bits       = 4;
        fprintf(stderr, "[config] WARNING: no quantization_config, defaulting to 4-bit group_size=64\n");
    }
    // Routing gate may have different quantization (e.g. MiniMax uses 8-bit gate)
    cfg.gate_bits = cfg.bits;
    cfg.gate_group_size = cfg.group_size;
    if (qc) {
        // Check for per-layer gate overrides: "model.layers.0.block_sparse_moe.gate" etc.
        NSString *gate0_key = [NSString stringWithFormat:@"model.layers.0.%s.gate", cfg.moe_prefix];
        NSDictionary *gate_qc = qc[gate0_key];
        if (gate_qc) {
            cfg.gate_bits = [gate_qc[@"bits"] intValue];
            cfg.gate_group_size = [gate_qc[@"group_size"] intValue];
            fprintf(stderr, "[config] Routing gate quantization: %d-bit, group_size=%d\n",
                    cfg.gate_bits, cfg.gate_group_size);
        }
    }

    // ---- RoPE parameters ----
    NSDictionary *rope = tc[@"rope_parameters"];
    if (rope) {
        // Qwen: nested rope_parameters
        cfg.rope_theta    = [rope[@"rope_theta"] floatValue];
        cfg.partial_rotary = [rope[@"partial_rotary_factor"] floatValue];
    } else if (tc[@"rope_theta"]) {
        // MiniMax: flat rope_theta + explicit rotary_dim
        cfg.rope_theta    = [tc[@"rope_theta"] floatValue];
        cfg.partial_rotary = 0.0f;  // not used when rotary_dim is explicit
    } else {
        cfg.rope_theta    = 10000000.0f;
        cfg.partial_rotary = 0.25f;
    }

    // ---- Layer types ----
    cfg.is_full_attn    = calloc(cfg.num_layers, sizeof(bool));
    cfg.full_attn_index = malloc(cfg.num_layers * sizeof(int));
    cfg.linear_index    = malloc(cfg.num_layers * sizeof(int));

    // Try explicit layer_types array (Qwen style: ["linear_attention", "full_attention", ...])
    NSArray *layerTypes = tc[@"layer_types"];
    // Also try attn_type_list (MiniMax style: [1, 1, 1, ...] where 1=full)
    NSArray *attnTypeList = tc[@"attn_type_list"] ?: root[@"attn_type_list"];

    if (layerTypes && [layerTypes count] == (NSUInteger)cfg.num_layers) {
        for (int i = 0; i < cfg.num_layers; i++) {
            cfg.is_full_attn[i] = [layerTypes[i] isEqualToString:@"full_attention"];
        }
    } else if (attnTypeList && [attnTypeList count] == (NSUInteger)cfg.num_layers) {
        for (int i = 0; i < cfg.num_layers; i++) {
            cfg.is_full_attn[i] = ([attnTypeList[i] intValue] == 1);
        }
    } else {
        int interval = tc[@"full_attention_interval"] ? [tc[@"full_attention_interval"] intValue] : 4;
        for (int i = 0; i < cfg.num_layers; i++) {
            cfg.is_full_attn[i] = ((i + 1) % interval == 0);
        }
        fprintf(stderr, "[config] Using full_attn_interval=%d (no explicit layer_types)\n", interval);
    }

    int full_count = 0, linear_count = 0;
    for (int i = 0; i < cfg.num_layers; i++) {
        if (cfg.is_full_attn[i]) {
            cfg.full_attn_index[i] = full_count++;
            cfg.linear_index[i] = -1;
        } else {
            cfg.linear_index[i] = linear_count++;
            cfg.full_attn_index[i] = -1;
        }
    }
    cfg.num_full_attn_layers = full_count;
    cfg.num_linear_layers = linear_count;

    // ---- EOS tokens (can be int or array in config.json) ----
    id eosVal = root[@"eos_token_id"];
    if ([eosVal isKindOfClass:[NSArray class]]) {
        NSArray *arr = (NSArray *)eosVal;
        cfg.num_eos_tokens = (int)[arr count];
        if (cfg.num_eos_tokens > 8) cfg.num_eos_tokens = 8;
        for (int i = 0; i < cfg.num_eos_tokens; i++)
            cfg.eos_token_ids[i] = [arr[i] intValue];
    } else if (eosVal) {
        cfg.num_eos_tokens = 1;
        cfg.eos_token_ids[0] = [eosVal intValue];
    }

    // ---- Think tokens from tokenizer.json added_tokens ----
    NSString *tokPath = [base stringByAppendingPathComponent:@"tokenizer.json"];
    if ([fm fileExistsAtPath:tokPath]) {
        NSData *tokData = [NSData dataWithContentsOfFile:tokPath];
        NSDictionary *tokRoot = [NSJSONSerialization JSONObjectWithData:tokData options:0 error:nil];
        NSArray *addedTokens = tokRoot[@"added_tokens"];
        if (addedTokens) {
            for (NSDictionary *tok in addedTokens) {
                NSString *content = tok[@"content"];
                int tid = [tok[@"id"] intValue];
                if ([content isEqualToString:@"<think>"]) cfg.think_start_token = tid;
                else if ([content isEqualToString:@"</think>"]) cfg.think_end_token = tid;
            }
        }
    } else {
        fprintf(stderr, "[config] WARNING: tokenizer.json not found, think tokens disabled\n");
    }

    // ---- Derived dimensions ----
    cfg.linear_total_key   = cfg.linear_num_k_heads * cfg.linear_key_dim;
    cfg.linear_total_value = cfg.linear_num_v_heads * cfg.linear_value_dim;
    cfg.linear_conv_dim    = cfg.linear_total_key * 2 + cfg.linear_total_value;

    // RoPE rotary_dim: explicit field takes priority, otherwise computed from partial_rotary
    if (tc[@"rotary_dim"]) {
        cfg.rotary_dim = [tc[@"rotary_dim"] intValue];
    } else {
        cfg.rotary_dim = (int)(cfg.head_dim * cfg.partial_rotary);
    }

    // Expert byte offsets
    compute_expert_offsets(&cfg);

    // ---- Summary ----
    fprintf(stderr, "[config] %d layers (%d linear + %d full), hidden=%d, heads=%d, kv_heads=%d, head_dim=%d\n",
            cfg.num_layers, cfg.num_linear_layers, cfg.num_full_attn_layers,
            cfg.hidden_dim, cfg.num_attn_heads, cfg.num_kv_heads, cfg.head_dim);
    fprintf(stderr, "[config] %d experts (K=%d), moe_intermediate=%d, shared=%d, routing=%s\n",
            cfg.num_experts, cfg.num_experts_per_tok, cfg.moe_intermediate, cfg.shared_intermediate,
            cfg.scoring_func ? "sigmoid" : "softmax");
    fprintf(stderr, "[config] %d-bit quantization, group_size=%d, expert_size=%zu bytes, rotary_dim=%d\n",
            cfg.bits, cfg.group_size, cfg.expert_size_4bit, cfg.rotary_dim);
    if (cfg.gate_bits != cfg.bits)
        fprintf(stderr, "[config] Routing gate: %d-bit, group_size=%d\n", cfg.gate_bits, cfg.gate_group_size);
    fprintf(stderr, "[config] attn_gate=%d, qk_norm_per_layer=%d, moe_prefix=%s\n",
            cfg.has_attn_gate, cfg.qk_norm_per_layer, cfg.moe_prefix);
    fprintf(stderr, "[config] EOS tokens: [");
    for (int i = 0; i < cfg.num_eos_tokens; i++)
        fprintf(stderr, "%s%d", i ? ", " : "", cfg.eos_token_ids[i]);
    fprintf(stderr, "], think: %d/%d\n", cfg.think_start_token, cfg.think_end_token);
}

// ============================================================================
// Tiered manifest loader
// ============================================================================

int load_tiered_manifest(const char *model_path) {
    char manifest_path[1024];
    snprintf(manifest_path, sizeof(manifest_path),
             "%s/packed_experts_tiered/tiered_manifest.json", model_path);

    NSData *data = [NSData dataWithContentsOfFile:
        [NSString stringWithUTF8String:manifest_path]];
    if (!data) return 0;  // No tiered manifest found

    NSError *err = nil;
    NSDictionary *root = [NSJSONSerialization JSONObjectWithData:data options:0 error:&err];
    if (!root || err) {
        fprintf(stderr, "[tiered] Failed to parse %s: %s\n",
                manifest_path, [[err localizedDescription] UTF8String]);
        return 0;
    }

    int num_layers = [root[@"num_layers"] intValue];
    int num_experts = [root[@"num_experts"] intValue];

    if (num_layers != cfg.num_layers || num_experts != cfg.num_experts) {
        fprintf(stderr, "[tiered] Manifest mismatch: %dx%d vs config %dx%d\n",
                num_layers, num_experts, cfg.num_layers, cfg.num_experts);
        return 0;
    }

    // Validate manifest expert sizes match runtime computation
    size_t manifest_size_4 = [root[@"expert_size_4bit"] unsignedLongLongValue];
    size_t manifest_size_2 = [root[@"expert_size_2bit"] unsignedLongLongValue];
    if (manifest_size_4 && manifest_size_4 != cfg.expert_size_4bit) {
        fprintf(stderr, "[tiered] expert_size_4bit mismatch: manifest=%zu vs config=%zu\n",
                manifest_size_4, cfg.expert_size_4bit);
        return 0;
    }
    if (manifest_size_2 && manifest_size_2 != cfg.expert_size_2bit) {
        fprintf(stderr, "[tiered] expert_size_2bit mismatch: manifest=%zu vs config=%zu\n",
                manifest_size_2, cfg.expert_size_2bit);
        return 0;
    }

    g_tiered_manifest = calloc(num_layers * num_experts, sizeof(TieredExpertInfo));

    NSDictionary *layers = root[@"layers"];
    int errors = 0;
    for (int l = 0; l < num_layers; l++) {
        NSString *lkey = [NSString stringWithFormat:@"%d", l];
        NSDictionary *layer = layers[lkey];
        if (!layer) {
            // Missing layer: default all experts to 4-bit with sequential offsets
            fprintf(stderr, "[tiered] WARNING: layer %d missing from manifest, defaulting to 4-bit\n", l);
            for (int e = 0; e < num_experts; e++) {
                TIERED(l, e).offset = (size_t)e * cfg.expert_size_4bit;
                TIERED(l, e).size = cfg.expert_size_4bit;
                TIERED(l, e).bits = 4;
            }
            continue;
        }

        NSArray *experts = layer[@"experts"];
        for (int e = 0; e < num_experts && e < (int)[experts count]; e++) {
            NSDictionary *exp = experts[e];
            int bits = [exp[@"bits"] intValue];
            if (bits != 2 && bits != 4) {
                fprintf(stderr, "[tiered] ERROR: layer %d expert %d has invalid bits=%d\n", l, e, bits);
                errors++;
                bits = 4;  // fallback
            }
            TIERED(l, e).offset = [exp[@"offset"] unsignedLongLongValue];
            TIERED(l, e).size = [exp[@"size"] unsignedLongLongValue];
            TIERED(l, e).bits = bits;
        }
    }

    if (errors > 0) {
        fprintf(stderr, "[tiered] WARNING: %d invalid entries found in manifest\n", errors);
    }

    // Print summary
    int hot = 0, cold = 0;
    for (int l = 0; l < num_layers; l++) {
        for (int e = 0; e < num_experts; e++) {
            if (TIERED(l, e).bits == 4) hot++;
            else cold++;
        }
    }
    double threshold = [root[@"threshold"] doubleValue];
    printf("[tiered] Loaded manifest: %d hot (4-bit) + %d cold (2-bit), threshold=%.0f%%\n",
           hot, cold, threshold * 100);

    return 1;
}

// ============================================================================
// Dynamic tracking arrays (allocated after config is loaded)
// Declarations here, alloc_tracking_arrays() defined after types below.
// ============================================================================

static int *g_expert_freq = NULL;
static uint8_t *g_expert_seen = NULL;
static void **g_lz4_index = NULL;  // actually LZ4IndexEntry**, cast at use site
static uint8_t *g_cache_seen = NULL;
static uint64_t *g_cache_last_touch_token = NULL;
static uint64_t *g_cache_last_evict_token = NULL;
static int *g_pred_experts = NULL;
static int *g_pred_count = NULL;

// GPU KV cache sequence length — set from cfg.max_seq_len in metal_setup()
// Default 8192 for desktop, iOS overrides via max_seq_len cap
static int GPU_KV_SEQ = 8192;

// Helper macros for flattened 2D access
#define FREQ(l, e)           g_expert_freq[(l) * cfg.num_experts + (e)]
#define EXPERT_SEEN_BYTE(l, e) g_expert_seen[(l) * ((cfg.num_experts + 7) / 8) + ((e) >> 3)]
#define CACHE_SEEN(l, e)     g_cache_seen[(l) * cfg.num_experts + (e)]
#define CACHE_TOUCH(l, e)    g_cache_last_touch_token[(l) * cfg.num_experts + (e)]
#define CACHE_EVICT(l, e)    g_cache_last_evict_token[(l) * cfg.num_experts + (e)]
#define PRED_EXPERT(l, k)    g_pred_experts[(l) * MAX_K + (k)]
#define PRED_COUNT(l)        g_pred_count[(l)]

// Forward declaration — defined after LayerWeightCache and LZ4IndexEntry
void alloc_tracking_arrays(void);


// ============================================================================
// Timing helper
// ============================================================================

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// ============================================================================
// Per-phase timing accumulators for fused_layer_forward
// Tracks time spent in each pipeline phase across all layers per token.
// Reset at token boundary, printed as summary.
// ============================================================================

typedef struct {
    double deferred_wait;    // waiting for previous CMD3 GPU
    double deferred_cpu;     // CPU readback + combine for deferred experts
    double input_norm;       // CPU RMS norm + CMD1 prep
    double cmd1_submit;      // CMD1 encode + commit
    double cmd1_wait;        // CMD1 waitUntilCompleted
    double cpu_attn;         // CPU attention compute (delta-net or full-attn)
    double cmd2_encode;      // CMD2 encode (o_proj + residual + norm + routing)
    double cmd2_wait;        // CMD2 commit + waitUntilCompleted
    double routing_cpu;      // CPU softmax + topK
    double spec_route;       // speculative early routing (gate matvec + topK)
    double expert_io;        // parallel pread + cache lookup
    double cmd3_encode;      // CMD3 encode experts + submit (deferred)
    double total;            // total per-layer time
    int count;               // number of layers timed
} LayerTimingAccum;

static LayerTimingAccum g_timing = {0};
static int g_timing_enabled = 0;

// Temporal prediction pipeline counters (declared early for timing_print access)
static int g_pred_enabled = 0;
static int g_pred_generating = 0;   // only set to 1 after prefill (predictions only help during generation)
static uint64_t g_pred_hits = 0;
static uint64_t g_pred_misses = 0;
static uint64_t g_pred_layers = 0;

// Routing data collection for training an expert predictor
// Binary format per sample: int32 layer_idx, int32 K, float32[4096] hidden, int32[K] expert_indices
static FILE *g_routing_log = NULL;
static int g_routing_log_samples = 0;

// LZ4 compressed expert support
// File format: [LZ4IndexEntry × 512] + [compressed blobs]
typedef struct {
    uint64_t offset;
    uint32_t comp_size;
    uint32_t raw_size;
} LZ4IndexEntry;

static void *g_lz4_comp_bufs[16];                 // pre-allocated compressed read buffers (matches MAX_K)
static int g_use_lz4 = 0;                        // auto-detected from packed_experts_lz4/

// ============================================================================
// Expert frequency tracking (diagnostic: --freq flag)
// ============================================================================

static int g_freq_tracking = 0;  // enabled by --freq flag
static int g_use_2bit = 0;       // enabled by --2bit flag: use packed_experts_2bit/ + 2-bit kernel
static int g_cache_telemetry_enabled = 0;  // enabled by --cache-telemetry flag
static int g_cache_io_split = 1;  // >1: split each routed expert pread into N page-aligned chunks (fanout)
static int g_think_budget = 2048; // max thinking tokens before force-emitting </think>

// ---- Optimization flags (ported from develop branch) ----
static int g_fused_expert_enabled = 0;   // 1: fused gate+up+SwiGLU kernel, 0: separate dispatches (default OFF until validated)
static int g_cmd_merge_enabled = 1;      // 1: merge CMD1+CMD2 for linear attention (saves ~2ms/token), 0: separate
static int g_fused_attention_enabled = 0; // 1: fused online softmax attention (experimental), 0: 3-kernel fallback
static int g_use_fp16_accum = 0;         // 1: use half-precision accumulation in dequant kernels (experimental)
static int g_use_fp8_kv = 0;            // 1: use FP8 E4M3 KV cache (4x memory reduction)
static int g_sliding_window = 0;        // >0: sliding window size for full attention KV cache (circular buffer)
static int g_h2o_budget = 0;            // 0 = disabled, >0 = total H2O KV cache budget (sinks + recent + heavy hitters)
static int g_h2o_num_sinks = 4;         // attention sink tokens to keep (first N, typically 4)
static int g_expert_prefetch_enabled = 0; // 1: cross-layer expert prefetch (overlaps I/O with GPU compute)
static int *g_expert_prefetch_layer_fds = NULL; // [num_layers] fds for cross-layer prefetch (set by infer_init)
static float g_merged_shared_gate_score = 0.0f; // CMD1+CMD2 merge: carries shared gate score across phases
static int g_prefetch_hits_total = 0;    // expert prefetch diagnostic counters
static int g_prefetch_misses_total = 0;

// ---- FP8 E4M3 encode/decode (inline, for KV cache quantization) ----
// FP8 E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits
// Exponent bias: 7.  Range: [-448, 448].  NaN: 0x7F.

static inline uint8_t fp8_e4m3_encode(float x, float inv_scale) {
    float scaled = x * inv_scale;
    if (scaled != scaled) return 0x7F;  // NaN -> FP8 NaN
    scaled = fminf(fmaxf(scaled, -448.0f), 448.0f);

    uint32_t bits;
    memcpy(&bits, &scaled, sizeof(bits));
    uint8_t sign = (bits >> 31) & 1;
    float mag = fabsf(scaled);

    if (mag < 0.001953125f) {
        int m = (int)(mag / 0.001953125f + 0.5f);
        if (m > 7) m = 7;
        return (uint8_t)((sign << 7) | m);
    }

    int exp_unbiased = (int)floorf(log2f(mag));
    if (exp_unbiased < -6) exp_unbiased = -6;
    if (exp_unbiased > 8)  exp_unbiased = 8;

    float frac = mag / powf(2.0f, (float)exp_unbiased) - 1.0f;
    int mantissa = (int)(frac * 8.0f + 0.5f);
    if (mantissa > 7) { mantissa = 0; exp_unbiased++; }

    int exp_biased = exp_unbiased + 7;
    if (exp_biased < 0) { exp_biased = 0; mantissa = 0; }
    if (exp_biased > 15) { exp_biased = 15; mantissa = 6; }
    if (exp_biased == 15 && mantissa >= 7) mantissa = 6;

    return (uint8_t)((sign << 7) | (exp_biased << 3) | mantissa);
}

static inline float fp8_e4m3_decode(uint8_t x, float scale) {
    if (x == 0x7F) return __builtin_nanf("");
    uint8_t sign = (x >> 7) & 1;
    uint8_t exp_biased = (x >> 3) & 0xF;
    uint8_t mantissa = x & 0x7;
    float val;
    if (exp_biased == 0) {
        val = (float)mantissa * 0.001953125f;
    } else {
        val = (1.0f + (float)mantissa / 8.0f) * powf(2.0f, (float)exp_biased - 7.0f);
    }
    if (sign) val = -val;
    return val * scale;
}

static inline float fp8_absmax(const float *vec, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(vec[i]);
        if (a > amax) amax = a;
    }
    return amax;
}

static inline float fp8_encode_vec(const float *src, uint8_t *dst, int n) {
    float amax = fp8_absmax(src, n);
    float scale = (amax > 0.0f) ? (amax / 240.0f) : 1.0f;
    float inv_scale = 1.0f / scale;
    for (int i = 0; i < n; i++) {
        dst[i] = fp8_e4m3_encode(src[i], inv_scale);
    }
    return scale;
}

static inline void fp8_decode_vec(const uint8_t *src, float *dst, int n, float scale) {
    for (int i = 0; i < n; i++) {
        dst[i] = fp8_e4m3_decode(src[i], scale);
    }
}
static int g_use_q3_experts = 0;         // enabled by --q3-experts flag: use packed_experts_Q3/ with exact GGUF routed experts
static int g_layer_is_q3_hybrid[MAX_LAYERS];  // per-layer quant: 1=Q3 hybrid, 0=4-bit
static int g_use_q3_outlier = 0;  // active layer override: exact layer-27 IQ4_XS gate/up + Q5_K down
static int g_layer_is_q3_outlier[MAX_LAYERS];
static int g_layer_is_2bit[MAX_LAYERS];  // per-layer quant: 1=2-bit, 0=4-bit (for mixed quant)
static int g_stream_mode = 0;    // --stream: clean output only, no progress/stats
static int g_nax_disabled = 1;   // NAX disabled by default (slower for M=1 decode); --nax to enable
static int g_nax_min_batch = 4;  // minimum batch size to use NAX (M=1 wastes 31/32 of tile)

// ---- Prefill batching ----
static int g_prefill_batch = 1;  // --pfb N: batch N tokens per layer during prefill (default 1 = no batching)
static int g_prefill_skip_experts = 0; // --prefill-skip-experts: skip routed expert I/O during intermediate prefill tokens
static int g_prefill_k = -1;  // --prefill-k N: override K for intermediate prefill tokens (-1 = use default)
static int g_prefill_experts_full_only = 0; // --prefill-experts-full-only: K=0 for linear layers, full K for full-attn layers
#define MAX_PFB 256              // maximum prefill batch size
#define MAX_PFB_GPU 32           // FMA kernel accumulator limit (float acc[32])
#define MAX_PFB_NAX 128          // NAX kernel has no static limit; 128 balances memory vs throughput

// ---- Prefill abort ----
static atomic_int g_prefill_abort = 0;  // set to 1 to abort prefill early

// ---- Optimization toggles (for A/B profiling) ----
static int g_disable_gpu_combine = 0;    // disable fused CMD3 combine+residual+norm on GPU
static int g_disable_fused_experts = 0;  // disable batched expert GPU encoding (fall back to sequential)
static int g_disable_expert_prefetch = 0;// disable async pread prefetch (use synchronous pread)
static int g_disable_batched_linear = 0; // set to 1 to disable batched linear prefill (A/B testing)
// gpu_linear_attn_enabled already exists (line ~4629) for fused attention toggle

static inline int effective_prefill_skip_experts(void) {
    return g_prefill_skip_experts;
}

// Tiered I/O: cold fds (F_NOCACHE) for first reads, warm fds (page cached) for repeats
static int *g_layer_fds_cold = NULL;    // [cfg.num_layers] cold fds (set in main)

// Async pread state defined after InferPreadTask (see below)

static inline int expert_is_seen(int layer, int expert) {
    return (EXPERT_SEEN_BYTE(layer, expert) >> (expert & 7)) & 1;
}
static inline void expert_mark_seen(int layer, int expert) {
    EXPERT_SEEN_BYTE(layer, expert) |= (1 << (expert & 7));
}
// Pick fd for expert read. Currently: always use warm fd (OS page cache).
// Tiered I/O (cold F_NOCACHE for first reads) was tested but OS page cache
// without any bypass outperforms all custom caching strategies.
static inline int expert_pick_fd(int layer, int expert, int warm_fd) {
    (void)layer; (void)expert;
    return warm_fd;
}

typedef enum {
    EXPERT_QUANT_4BIT = 0,
    EXPERT_QUANT_2BIT = 1,
    EXPERT_QUANT_Q3_HYBRID = 2,
    EXPERT_QUANT_Q3_OUTLIER = 3,
} ExpertQuantKind;

typedef enum {
    EXPERT_PROJ_AFFINE = 0,
    EXPERT_PROJ_IQ3_XXS = 1,
    EXPERT_PROJ_IQ4_XS = 2,
    EXPERT_PROJ_Q5_K = 3,
} ExpertProjectionKind;

typedef struct {
    size_t expert_size;
    NSUInteger gate_w_off, gate_s_off, gate_b_off;
    NSUInteger up_w_off, up_s_off, up_b_off;
    NSUInteger down_w_off, down_s_off, down_b_off;
    ExpertProjectionKind gate_kind;
    ExpertProjectionKind up_kind;
    ExpertProjectionKind down_kind;
} ExpertLayout;

static ExpertLayout g_q3_layer_layouts[MAX_LAYERS];
static int g_q3_layer_layout_valid[MAX_LAYERS];
static int g_q3_layout_manifest_loaded = 0;
static ExpertLayout g_active_q3_layout;
static int g_active_q3_layout_valid = 0;

static inline ExpertProjectionKind expert_projection_kind_from_quant_name(const char *quant_name) {
    if (!quant_name) return EXPERT_PROJ_AFFINE;
    if (strcmp(quant_name, "IQ3_XXS") == 0) return EXPERT_PROJ_IQ3_XXS;
    if (strcmp(quant_name, "IQ4_XS") == 0) return EXPERT_PROJ_IQ4_XS;
    if (strcmp(quant_name, "Q5_K") == 0) return EXPERT_PROJ_Q5_K;
    return EXPERT_PROJ_AFFINE;
}

static inline int expert_layout_is_q3_outlier(const ExpertLayout *layout) {
    if (!layout) return 0;
    return layout->gate_kind == EXPERT_PROJ_IQ4_XS ||
           layout->up_kind == EXPERT_PROJ_IQ4_XS ||
           layout->down_kind == EXPERT_PROJ_Q5_K;
}

static inline ExpertQuantKind active_expert_quant_kind(void) {
    if (g_use_q3_outlier) return EXPERT_QUANT_Q3_OUTLIER;
    if (g_use_q3_experts) return EXPERT_QUANT_Q3_HYBRID;
    if (g_use_2bit) return EXPERT_QUANT_2BIT;
    return EXPERT_QUANT_4BIT;
}

static inline ExpertQuantKind layer_expert_quant_kind(int layer) {
    if (g_layer_is_q3_outlier[layer]) return EXPERT_QUANT_Q3_OUTLIER;
    if (g_layer_is_q3_hybrid[layer]) return EXPERT_QUANT_Q3_HYBRID;
    if (g_layer_is_2bit[layer]) return EXPERT_QUANT_2BIT;
    return EXPERT_QUANT_4BIT;
}

// Per-expert quant kind for tiered mode
static inline ExpertQuantKind tiered_expert_quant_kind(int layer, int expert) {
    if (g_use_tiered && g_tiered_manifest) {
        return (TIERED(layer, expert).bits == 2) ? EXPERT_QUANT_2BIT : EXPERT_QUANT_4BIT;
    }
    return layer_expert_quant_kind(layer);
}

static int parse_q3_layout_components(NSArray *components, size_t expert_size, ExpertLayout *layout_out) {
    if (!components || !layout_out) return 0;
    ExpertLayout layout = {0};
    layout.expert_size = expert_size;
    int have_gate = 0, have_up = 0, have_down = 0;

    for (NSDictionary *comp in components) {
        NSString *name = comp[@"name"];
        NSString *quant = comp[@"quant"];
        NSNumber *offset_num = comp[@"offset"];
        if (!name || !quant || !offset_num) continue;
        NSUInteger off = (NSUInteger)[offset_num unsignedLongLongValue];
        ExpertProjectionKind kind = expert_projection_kind_from_quant_name([quant UTF8String]);
        if ([name isEqualToString:@"gate_proj.weight"]) {
            layout.gate_w_off = off;
            layout.gate_s_off = off;
            layout.gate_b_off = off;
            layout.gate_kind = kind;
            have_gate = 1;
        } else if ([name isEqualToString:@"up_proj.weight"]) {
            layout.up_w_off = off;
            layout.up_s_off = off;
            layout.up_b_off = off;
            layout.up_kind = kind;
            have_up = 1;
        } else if ([name isEqualToString:@"down_proj.weight"]) {
            layout.down_w_off = off;
            layout.down_s_off = off;
            layout.down_b_off = off;
            layout.down_kind = kind;
            have_down = 1;
        }
    }

    if (!have_gate || !have_up || !have_down) return 0;
    *layout_out = layout;
    return 1;
}

static int load_q3_layout_manifest(const char *model_path) {
    @autoreleasepool {
        memset(g_q3_layer_layout_valid, 0, sizeof(g_q3_layer_layout_valid));
        g_q3_layout_manifest_loaded = 0;

        char manifest_path[1024];
        snprintf(manifest_path, sizeof(manifest_path),
                 "%s/packed_experts_Q3/layout.json", model_path);

        NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:manifest_path]];
        if (!data) return 0;

        NSError *err = nil;
        NSDictionary *root = [NSJSONSerialization JSONObjectWithData:data options:0 error:&err];
        if (!root || err) {
            fprintf(stderr, "[q3] Failed to parse %s\n", manifest_path);
            return 0;
        }

        NSNumber *num_layers_num = root[@"num_layers"];
        NSNumber *num_experts_num = root[@"num_experts"];
        if (num_layers_num && [num_layers_num intValue] != cfg.num_layers) {
            fprintf(stderr, "[q3] layout.json num_layers=%d vs config %d\n",
                    [num_layers_num intValue], cfg.num_layers);
            return 0;
        }
        if (num_experts_num && [num_experts_num intValue] != cfg.num_experts) {
            fprintf(stderr, "[q3] layout.json num_experts=%d vs config %d\n",
                    [num_experts_num intValue], cfg.num_experts);
            return 0;
        }

        NSDictionary *layers = root[@"layers"];
        if ([layers isKindOfClass:[NSDictionary class]] && [layers count] > 0) {
            for (NSString *layer_key in layers) {
                int layer = [layer_key intValue];
                if (layer < 0 || layer >= cfg.num_layers) continue;
                NSDictionary *layer_info = layers[layer_key];
                NSNumber *expert_size_num = layer_info[@"expert_size"];
                NSArray *components = layer_info[@"components"];
                if (!expert_size_num || !components) continue;
                ExpertLayout layout = {0};
                if (!parse_q3_layout_components(components, (size_t)[expert_size_num unsignedLongLongValue], &layout)) {
                    fprintf(stderr, "[q3] Invalid components in layer %d manifest entry\n", layer);
                    return 0;
                }
                g_q3_layer_layouts[layer] = layout;
                g_q3_layer_layout_valid[layer] = 1;
            }
            g_q3_layout_manifest_loaded = 1;
            return 1;
        }

        // Backward-compatible parser for the original 397B manifest shape.
        NSNumber *expert_size_num = root[@"expert_size"];
        NSArray *components = root[@"components"];
        if (!expert_size_num || !components) return 0;

        ExpertLayout default_layout = {0};
        if (!parse_q3_layout_components(components, (size_t)[expert_size_num unsignedLongLongValue], &default_layout)) {
            fprintf(stderr, "[q3] Invalid default components in %s\n", manifest_path);
            return 0;
        }
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            g_q3_layer_layouts[layer] = default_layout;
            g_q3_layer_layout_valid[layer] = 1;
        }

        NSDictionary *outlier_layers = root[@"outlier_layers"];
        if ([outlier_layers isKindOfClass:[NSDictionary class]]) {
            for (NSString *layer_key in outlier_layers) {
                int layer = [layer_key intValue];
                if (layer < 0 || layer >= cfg.num_layers) continue;
                NSDictionary *layer_info = outlier_layers[layer_key];
                NSNumber *layer_expert_size_num = layer_info[@"expert_size"];
                NSArray *layer_components = layer_info[@"components"];
                if (!layer_expert_size_num || !layer_components) continue;
                ExpertLayout outlier_layout = {0};
                if (!parse_q3_layout_components(layer_components, (size_t)[layer_expert_size_num unsignedLongLongValue], &outlier_layout)) {
                    fprintf(stderr, "[q3] Invalid outlier components for layer %d\n", layer);
                    return 0;
                }
                g_q3_layer_layouts[layer] = outlier_layout;
                g_q3_layer_layout_valid[layer] = 1;
            }
        }

        g_q3_layout_manifest_loaded = 1;
        return 1;
    }
}

static inline size_t default_q3_expert_size(void) {
    if (g_q3_layout_manifest_loaded) {
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            if (g_q3_layer_layout_valid[layer]) {
                return g_q3_layer_layouts[layer].expert_size;
            }
        }
    }
    return g_use_q3_outlier ? EXPERT_SIZE_Q3_OUTLIER : EXPERT_SIZE_Q3_HYBRID;
}

static inline size_t max_q3_expert_size(void) {
    size_t max_size = 0;
    if (g_q3_layout_manifest_loaded) {
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            if (g_q3_layer_layout_valid[layer] && g_q3_layer_layouts[layer].expert_size > max_size) {
                max_size = g_q3_layer_layouts[layer].expert_size;
            }
        }
    }
    return max_size ? max_size : EXPERT_SIZE_Q3_OUTLIER;
}

static inline ExpertLayout expert_layout_for_kind(ExpertQuantKind kind) {
    switch (kind) {
        case EXPERT_QUANT_2BIT:
            return (ExpertLayout) {
                .expert_size = EXPERT_SIZE_2BIT,
                .gate_w_off = GATE_W_OFF_2, .gate_s_off = GATE_S_OFF_2, .gate_b_off = GATE_B_OFF_2,
                .up_w_off   = UP_W_OFF_2,   .up_s_off   = UP_S_OFF_2,   .up_b_off   = UP_B_OFF_2,
                .down_w_off = DOWN_W_OFF_2, .down_s_off = DOWN_S_OFF_2, .down_b_off = DOWN_B_OFF_2,
                .gate_kind = EXPERT_PROJ_AFFINE,
                .up_kind = EXPERT_PROJ_AFFINE,
                .down_kind = EXPERT_PROJ_AFFINE,
            };
        case EXPERT_QUANT_Q3_HYBRID:
            if (g_active_q3_layout_valid) return g_active_q3_layout;
            return (ExpertLayout) {
                .expert_size = EXPERT_SIZE_Q3_HYBRID,
                .gate_w_off = GATE_W_OFF_Q3, .gate_s_off = GATE_W_OFF_Q3, .gate_b_off = GATE_W_OFF_Q3,
                .up_w_off   = UP_W_OFF_Q3,   .up_s_off   = UP_W_OFF_Q3,   .up_b_off   = UP_W_OFF_Q3,
                .down_w_off = DOWN_W_OFF_Q3, .down_s_off = DOWN_S_OFF_Q3, .down_b_off = DOWN_B_OFF_Q3,
                .gate_kind = EXPERT_PROJ_IQ3_XXS,
                .up_kind = EXPERT_PROJ_IQ3_XXS,
                .down_kind = EXPERT_PROJ_IQ4_XS,
            };
        case EXPERT_QUANT_Q3_OUTLIER:
            if (g_active_q3_layout_valid) return g_active_q3_layout;
            return (ExpertLayout) {
                .expert_size = EXPERT_SIZE_Q3_OUTLIER,
                .gate_w_off = GATE_W_OFF_Q3_OUTLIER, .gate_s_off = GATE_W_OFF_Q3_OUTLIER, .gate_b_off = GATE_W_OFF_Q3_OUTLIER,
                .up_w_off   = UP_W_OFF_Q3_OUTLIER,   .up_s_off   = UP_W_OFF_Q3_OUTLIER,   .up_b_off   = UP_W_OFF_Q3_OUTLIER,
                .down_w_off = DOWN_W_OFF_Q3_OUTLIER, .down_s_off = DOWN_S_OFF_Q3_OUTLIER, .down_b_off = DOWN_B_OFF_Q3_OUTLIER,
                .gate_kind = EXPERT_PROJ_IQ4_XS,
                .up_kind = EXPERT_PROJ_IQ4_XS,
                .down_kind = EXPERT_PROJ_Q5_K,
            };
        case EXPERT_QUANT_4BIT:
        default:
            return (ExpertLayout) {
                .expert_size = EXPERT_SIZE,
                .gate_w_off = (int)cfg.gate_w_off_4, .gate_s_off = (int)cfg.gate_s_off_4, .gate_b_off = (int)cfg.gate_b_off_4,
                .up_w_off   = (int)cfg.up_w_off_4,   .up_s_off   = (int)cfg.up_s_off_4,   .up_b_off   = (int)cfg.up_b_off_4,
                .down_w_off = (int)cfg.down_w_off_4,  .down_s_off = (int)cfg.down_s_off_4,  .down_b_off = (int)cfg.down_b_off_4,
                .gate_kind = EXPERT_PROJ_AFFINE,
                .up_kind = EXPERT_PROJ_AFFINE,
                .down_kind = EXPERT_PROJ_AFFINE,
            };
    }
}

static inline const char *expert_quant_label(ExpertQuantKind kind) {
    switch (kind) {
        case EXPERT_QUANT_2BIT: return "2-bit";
        case EXPERT_QUANT_Q3_HYBRID: return "Q3-GGUF";
        case EXPERT_QUANT_Q3_OUTLIER: return "Q3-outlier";
        case EXPERT_QUANT_4BIT:
        default: return "4-bit";
    }
}

static inline const char *requested_expert_quant_label(void) {
    if (g_use_tiered) return "tiered (4/2-bit)";
    if (g_use_q3_experts) return "Q3-GGUF";
    if (g_use_2bit) return "2-bit";
    return "4-bit";
}

// Active expert size based on quantization mode
static inline size_t active_expert_size(void) {
    if ((g_use_q3_experts || g_use_q3_outlier) && g_active_q3_layout_valid) {
        return g_active_q3_layout.expert_size;
    }
    if ((g_use_q3_experts || g_use_q3_outlier) && g_q3_layout_manifest_loaded) {
        return default_q3_expert_size();
    }
    return expert_layout_for_kind(active_expert_quant_kind()).expert_size;
}

// Tiered-aware expert offset and size lookup
static inline void expert_offset_size(int layer, int expert, off_t *out_offset, size_t *out_size) {
    if (g_use_tiered && g_tiered_manifest) {
        TieredExpertInfo *ti = &TIERED(layer, expert);
        *out_offset = (off_t)ti->offset;
        *out_size = ti->size;
    } else {
        size_t esz = active_expert_size();
        *out_offset = (off_t)expert * esz;
        *out_size = esz;
    }
}

static inline size_t layer_expert_size(int layer) {
    if ((g_layer_is_q3_hybrid[layer] || g_layer_is_q3_outlier[layer]) &&
        g_q3_layout_manifest_loaded && g_q3_layer_layout_valid[layer]) {
        return g_q3_layer_layouts[layer].expert_size;
    }
    return expert_layout_for_kind(layer_expert_quant_kind(layer)).expert_size;
}

static inline size_t max_expert_size_for_current_config(void) {
    if (g_use_q3_experts) return max_q3_expert_size();
    return EXPERT_SIZE;
}
static int g_freq_total_tokens = 0;  // total tokens processed while tracking

typedef struct {
    uint64_t token_clock;
    uint64_t unique_experts_touched;
    uint64_t cold_misses;
    uint64_t eviction_misses;
    uint64_t evictions;
    uint64_t reuse_le_1;
    uint64_t reuse_le_4;
    uint64_t reuse_le_16;
    uint64_t reuse_le_64;
    uint64_t reuse_gt_64;
    uint64_t reuse_distance_sum;
    uint64_t reuse_distance_samples;
} CacheTelemetry;

static CacheTelemetry g_cache_telemetry = {0};

static void cache_telemetry_reset(void) {
    memset(&g_cache_telemetry, 0, sizeof(g_cache_telemetry));
    memset(g_cache_seen, 0, cfg.num_layers * cfg.num_experts * sizeof(uint8_t));
    memset(g_cache_last_touch_token, 0, cfg.num_layers * cfg.num_experts * sizeof(uint64_t));
    memset(g_cache_last_evict_token, 0, cfg.num_layers * cfg.num_experts * sizeof(uint64_t));
}

static void cache_telemetry_note_token(void) {
    if (!g_cache_telemetry_enabled) return;
    g_cache_telemetry.token_clock++;
}

static void cache_telemetry_touch(int layer_idx, int expert_idx) {
    if (!g_cache_telemetry_enabled) return;
    if (layer_idx < 0 || layer_idx >= cfg.num_layers || expert_idx < 0 || expert_idx >= cfg.num_experts) return;
    if (!CACHE_SEEN(layer_idx, expert_idx)) {
        CACHE_SEEN(layer_idx, expert_idx) = 1;
        g_cache_telemetry.unique_experts_touched++;
    }
    CACHE_TOUCH(layer_idx, expert_idx) = g_cache_telemetry.token_clock;
}

static void cache_telemetry_miss(int layer_idx, int expert_idx) {
    if (!g_cache_telemetry_enabled) return;
    if (layer_idx < 0 || layer_idx >= cfg.num_layers || expert_idx < 0 || expert_idx >= cfg.num_experts) return;
    if (!CACHE_SEEN(layer_idx, expert_idx)) {
        g_cache_telemetry.cold_misses++;
        CACHE_SEEN(layer_idx, expert_idx) = 1;
        g_cache_telemetry.unique_experts_touched++;
    } else {
        g_cache_telemetry.eviction_misses++;
        uint64_t dist = 0;
        if (CACHE_EVICT(layer_idx, expert_idx) > 0 &&
            g_cache_telemetry.token_clock >= CACHE_EVICT(layer_idx, expert_idx)) {
            dist = g_cache_telemetry.token_clock - CACHE_EVICT(layer_idx, expert_idx);
        }
        if (dist <= 1) g_cache_telemetry.reuse_le_1++;
        else if (dist <= 4) g_cache_telemetry.reuse_le_4++;
        else if (dist <= 16) g_cache_telemetry.reuse_le_16++;
        else if (dist <= 64) g_cache_telemetry.reuse_le_64++;
        else g_cache_telemetry.reuse_gt_64++;
        g_cache_telemetry.reuse_distance_sum += dist;
        g_cache_telemetry.reuse_distance_samples++;
    }
    CACHE_TOUCH(layer_idx, expert_idx) = g_cache_telemetry.token_clock;
}

static void cache_telemetry_evict(int layer_idx, int expert_idx) {
    if (!g_cache_telemetry_enabled) return;
    if (layer_idx < 0 || layer_idx >= cfg.num_layers || expert_idx < 0 || expert_idx >= cfg.num_experts) return;
    g_cache_telemetry.evictions++;
    CACHE_EVICT(layer_idx, expert_idx) = g_cache_telemetry.token_clock;
}

static void cache_telemetry_print(uint64_t hits, uint64_t misses) {
    if (!g_cache_telemetry_enabled) return;
    uint64_t total = hits + misses;
    fprintf(stderr, "\n=== Cache Telemetry ===\n");
    fprintf(stderr, "Tokens tracked: %llu\n", g_cache_telemetry.token_clock);
    fprintf(stderr, "Unique experts touched: %llu / %d (%.1f%%)\n",
            g_cache_telemetry.unique_experts_touched,
            cfg.num_layers * cfg.num_experts,
            100.0 * g_cache_telemetry.unique_experts_touched / (cfg.num_layers * cfg.num_experts));
    fprintf(stderr, "Miss breakdown: cold %llu (%.1f%% of misses), eviction %llu (%.1f%% of misses)\n",
            g_cache_telemetry.cold_misses,
            misses > 0 ? 100.0 * g_cache_telemetry.cold_misses / misses : 0.0,
            g_cache_telemetry.eviction_misses,
            misses > 0 ? 100.0 * g_cache_telemetry.eviction_misses / misses : 0.0);
    fprintf(stderr, "Evictions: %llu\n", g_cache_telemetry.evictions);
    fprintf(stderr, "Eviction reuse distance: <=1 tok %llu, <=4 %llu, <=16 %llu, <=64 %llu, >64 %llu",
            g_cache_telemetry.reuse_le_1,
            g_cache_telemetry.reuse_le_4,
            g_cache_telemetry.reuse_le_16,
            g_cache_telemetry.reuse_le_64,
            g_cache_telemetry.reuse_gt_64);
    if (g_cache_telemetry.reuse_distance_samples > 0) {
        fprintf(stderr, " (avg %.1f tok)\n",
                (double)g_cache_telemetry.reuse_distance_sum / g_cache_telemetry.reuse_distance_samples);
    } else {
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "Effective hit rate: %.1f%%\n",
            total > 0 ? 100.0 * hits / total : 0.0);
}

static void timing_reset(void) {
    memset(&g_timing, 0, sizeof(g_timing));
}

static void timing_print(void) {
    if (g_timing.count == 0) return;
    int n = g_timing.count;
    fprintf(stderr, "\n[timing] Per-layer breakdown (avg of %d layers, ms):\n", n);
    fprintf(stderr, "  deferred_wait:  %6.3f\n", g_timing.deferred_wait / n);
    fprintf(stderr, "  deferred_cpu:   %6.3f\n", g_timing.deferred_cpu / n);
    fprintf(stderr, "  input_norm:     %6.3f\n", g_timing.input_norm / n);
    fprintf(stderr, "  cmd1_submit:    %6.3f\n", g_timing.cmd1_submit / n);
    fprintf(stderr, "  cmd1_wait:      %6.3f\n", g_timing.cmd1_wait / n);
    fprintf(stderr, "  spec_route:     %6.3f\n", g_timing.spec_route / n);
    fprintf(stderr, "  cpu_attn:       %6.3f\n", g_timing.cpu_attn / n);
    fprintf(stderr, "  cmd2_encode:    %6.3f\n", g_timing.cmd2_encode / n);
    fprintf(stderr, "  cmd2_wait:      %6.3f\n", g_timing.cmd2_wait / n);
    fprintf(stderr, "  routing_cpu:    %6.3f\n", g_timing.routing_cpu / n);
    fprintf(stderr, "  expert_io:      %6.3f\n", g_timing.expert_io / n);
    fprintf(stderr, "  cmd3_encode:    %6.3f\n", g_timing.cmd3_encode / n);
    fprintf(stderr, "  total_layer:    %6.3f\n", g_timing.total / n);
    fprintf(stderr, "  sum_phases:     %6.3f\n",
            (g_timing.deferred_wait + g_timing.deferred_cpu + g_timing.input_norm +
             g_timing.cmd1_submit + g_timing.cmd1_wait + g_timing.spec_route +
             g_timing.cpu_attn +
             g_timing.cmd2_encode + g_timing.cmd2_wait + g_timing.routing_cpu +
             g_timing.expert_io + g_timing.cmd3_encode) / n);
    fprintf(stderr, "  cmd_buffers:    %d (3 per layer: CMD1+CMD2+CMD3)\n", n * 3);
    fprintf(stderr, "  sync_waits:     %d (2 per layer: CMD1+CMD2, CMD3 deferred)\n", n * 2);
    fprintf(stderr, "  gpu_encoders:   ~%d per layer (CMD1:3-4, CMD2:8-12, CMD3:~10)\n",
            22);  // approximate
    if (g_pred_enabled && g_pred_layers > 0) {
        uint64_t total = g_pred_hits + g_pred_misses;
        double hit_rate = total > 0 ? (double)g_pred_hits / total * 100.0 : 0;
        fprintf(stderr, "  [predict] hits=%llu misses=%llu rate=%.1f%% layers=%llu\n",
                g_pred_hits, g_pred_misses, hit_rate, g_pred_layers);
    }
}

// ============================================================================
// bf16 <-> f32 conversion (CPU side)
// ============================================================================

static float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

__attribute__((unused))
static uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}

// ============================================================================
// JSON parser (minimal, for model_weights.json)
// ============================================================================

// We use NSJSONSerialization via ObjC since we already link Foundation

typedef struct {
    const char *name;
    size_t offset;
    size_t size;
    int ndim;
    int shape[4];
    char dtype[8];  // "U32", "BF16", "F32"
} TensorInfo;

typedef struct {
    TensorInfo *tensors;
    int num_tensors;
    int capacity;
} TensorManifest;

static TensorManifest *load_manifest(const char *json_path) {
    @autoreleasepool {
        NSData *data = [NSData dataWithContentsOfFile:
            [NSString stringWithUTF8String:json_path]];
        if (!data) {
            fprintf(stderr, "ERROR: Cannot read %s\n", json_path);
            return NULL;
        }

        NSError *error = nil;
        NSDictionary *root = [NSJSONSerialization JSONObjectWithData:data
                                                             options:0
                                                               error:&error];
        if (!root) {
            fprintf(stderr, "ERROR: JSON parse failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return NULL;
        }

        NSDictionary *tensors = root[@"tensors"];
        if (!tensors) {
            fprintf(stderr, "ERROR: No 'tensors' key in manifest\n");
            return NULL;
        }

        TensorManifest *m = calloc(1, sizeof(TensorManifest));
        m->capacity = (int)[tensors count] + 16;
        m->tensors = calloc(m->capacity, sizeof(TensorInfo));
        m->num_tensors = 0;

        for (NSString *key in tensors) {
            NSDictionary *info = tensors[key];
            TensorInfo *t = &m->tensors[m->num_tensors];

            const char *name = [key UTF8String];
            t->name = strdup(name);
            t->offset = [info[@"offset"] unsignedLongLongValue];
            t->size = [info[@"size"] unsignedLongLongValue];

            NSArray *shape = info[@"shape"];
            t->ndim = (int)[shape count];
            for (int i = 0; i < t->ndim && i < 4; i++) {
                t->shape[i] = [shape[i] intValue];
            }

            const char *dtype = [info[@"dtype"] UTF8String];
            strncpy(t->dtype, dtype, 7);

            m->num_tensors++;
        }

        printf("[manifest] Loaded %d tensors from %s\n", m->num_tensors, json_path);
        return m;
    }
}

// Hash table for O(1) tensor lookup (replaces O(N) linear scan).
// FNV-1a hash, open addressing with linear probing.
#define TENSOR_HT_SIZE 8192  // power of 2, > 4x num_tensors (2092)

typedef struct {
    const char *key;     // tensor name (pointer into TensorInfo)
    TensorInfo *value;   // pointer to tensor info
} TensorHTEntry;

static TensorHTEntry tensor_ht[TENSOR_HT_SIZE];
static int tensor_ht_built = 0;

static uint32_t fnv1a(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; s++) {
        h ^= (uint8_t)*s;
        h *= 16777619u;
    }
    return h;
}

static void build_tensor_ht(TensorManifest *m) {
    if (tensor_ht_built) return;
    memset(tensor_ht, 0, sizeof(tensor_ht));
    for (int i = 0; i < m->num_tensors; i++) {
        uint32_t idx = fnv1a(m->tensors[i].name) & (TENSOR_HT_SIZE - 1);
        while (tensor_ht[idx].key) {
            idx = (idx + 1) & (TENSOR_HT_SIZE - 1);
        }
        tensor_ht[idx].key = m->tensors[i].name;
        tensor_ht[idx].value = &m->tensors[i];
    }
    tensor_ht_built = 1;
}

static TensorInfo *find_tensor(TensorManifest *m, const char *name) {
    if (!tensor_ht_built) build_tensor_ht(m);
    uint32_t idx = fnv1a(name) & (TENSOR_HT_SIZE - 1);
    while (tensor_ht[idx].key) {
        if (strcmp(tensor_ht[idx].key, name) == 0) {
            return tensor_ht[idx].value;
        }
        idx = (idx + 1) & (TENSOR_HT_SIZE - 1);
    }
    return NULL;
}

// ============================================================================
// Weight file: mmap'd binary blob
// ============================================================================

typedef struct WeightFile_s {
    void *data;
    size_t size;
    TensorManifest *manifest;
} WeightFile;

WeightFile *open_weights(const char *bin_path, const char *json_path) {
    // mmap the binary file
    int fd = open(bin_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ERROR: Cannot open %s: %s\n", bin_path, strerror(errno));
        return NULL;
    }

    struct stat st;
    fstat(fd, &st);
    size_t size = st.st_size;

    void *data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) {
        fprintf(stderr, "ERROR: mmap failed: %s\n", strerror(errno));
        return NULL;
    }

    // Advise sequential access
    madvise(data, size, MADV_SEQUENTIAL);

    TensorManifest *manifest = load_manifest(json_path);
    if (!manifest) {
        munmap(data, size);
        return NULL;
    }

    WeightFile *wf = calloc(1, sizeof(WeightFile));
    wf->data = data;
    wf->size = size;
    wf->manifest = manifest;

    printf("[weights] mmap'd %.2f GB from %s\n", size / 1e9, bin_path);
    return wf;
}

void *get_tensor_ptr(WeightFile *wf, const char *name) {
    TensorInfo *t = find_tensor(wf->manifest, name);
    if (!t) {
        fprintf(stderr, "WARNING: tensor '%s' not found\n", name);
        return NULL;
    }
    return (char *)wf->data + t->offset;
}

static TensorInfo *get_tensor_info(WeightFile *wf, const char *name) {
    return find_tensor(wf->manifest, name);
}

// ============================================================================
// Vocabulary for token decoding
// ============================================================================

typedef struct Vocabulary_s {
    char **tokens;   // token_id -> UTF-8 string
    int *lengths;    // token_id -> byte length
    int num_tokens;
} Vocabulary;

// GPT-2 BPE byte decoder: convert BPE Unicode chars back to raw bytes.
//
// GPT-2 BPE maps each of the 256 possible byte values to a Unicode codepoint:
//   printable ASCII (0x21-0x7E) → themselves (single-byte UTF-8, no conversion)
//   high bytes (0xA1-0xAC, 0xAE-0xFF) → themselves as codepoints (2-byte UTF-8)
//   all other bytes → U+0100 + sequential offset
//
// The vocab stores token strings using these codepoints (UTF-8 encoded).
// This function reverses the mapping: decodes each codepoint back to its raw byte.
//
// Without this, non-ASCII text gets corrupted — e.g., Russian "Д" (UTF-8: 0xD0 0x94)
// is stored as two BPE codepoints: U+00D0 (identity-mapped byte 0xD0) and U+0194
// (remapped byte 0x94).  The decoder must emit raw bytes 0xD0 0x94, which together
// form the correct UTF-8 for "Д".

// Reverse lookup table: codepoint → original byte.  Built once, used by all decodes.
static uint8_t g_bpe_cp_to_byte[512];
static int     g_bpe_table_built = 0;

static void bpe_build_decode_table(void) {
    if (g_bpe_table_built) return;

    // Mark all entries as "not a BPE codepoint" (0xFF is a valid byte, so we use
    // a separate flag approach — but since every cp 0..511 maps to exactly one
    // byte, we just fill the table the same way the encoder builds it).
    memset(g_bpe_cp_to_byte, 0, sizeof(g_bpe_cp_to_byte));

    int n = 0;
    for (int b = 0; b < 256; b++) {
        uint32_t cp;
        if ((b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF))
            cp = (uint32_t)b;        // identity-mapped
        else
            cp = 256 + (uint32_t)n++; // offset-mapped
        if (cp < 512) g_bpe_cp_to_byte[cp] = (uint8_t)b;
    }
    g_bpe_table_built = 1;
}

// Check if a codepoint is part of the GPT-2 BPE byte mapping (cp < 512 and in range).
static inline int bpe_is_mapped_cp(unsigned int cp) {
    if (cp >= 0x21 && cp <= 0x7E) return 1;  // identity-mapped ASCII
    if ((cp >= 0xA1 && cp <= 0xAC) || (cp >= 0xAE && cp <= 0xFF)) return 1;  // identity-mapped high
    if (cp >= 0x100 && cp < 0x100 + 68) return 1;  // offset-mapped (68 non-identity bytes)
    return 0;
}

static int bpe_decode_inplace(char *s, int len) {
    bpe_build_decode_table();

    int out = 0;
    int i = 0;
    while (i < len) {
        unsigned char c = (unsigned char)s[i];
        if (c < 0x80) {
            // 1-byte UTF-8 (ASCII).  Identity-mapped BPE codepoints (0x21-0x7E)
            // decode to themselves, and anything outside that range (e.g. newline,
            // space) shouldn't appear in raw BPE tokens.  Just pass through.
            s[out++] = s[i++];
        } else if ((c & 0xE0) == 0xC0 && i + 1 < len) {
            // 2-byte UTF-8: codepoint U+0080 to U+07FF
            unsigned int cp = ((c & 0x1F) << 6) | ((unsigned char)s[i+1] & 0x3F);
            if (bpe_is_mapped_cp(cp)) {
                // GPT-2 BPE codepoint → emit the single raw byte it represents
                s[out++] = (char)g_bpe_cp_to_byte[cp];
            } else {
                // Not a BPE codepoint — keep the UTF-8 encoding as-is
                s[out++] = s[i];
                s[out++] = s[i+1];
            }
            i += 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < len) {
            // 3-byte UTF-8: these are never BPE byte codepoints — pass through
            s[out++] = s[i]; s[out++] = s[i+1]; s[out++] = s[i+2];
            i += 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < len) {
            // 4-byte UTF-8: pass through
            s[out++] = s[i]; s[out++] = s[i+1]; s[out++] = s[i+2]; s[out++] = s[i+3];
            i += 4;
        } else {
            s[out++] = s[i++];
        }
    }
    s[out] = '\0';
    return out;
}

Vocabulary *load_vocab(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open vocab %s\n", path);
        return NULL;
    }

    uint32_t num_entries, max_id;
    fread(&num_entries, 4, 1, f);
    fread(&max_id, 4, 1, f);

    Vocabulary *v = calloc(1, sizeof(Vocabulary));
    v->num_tokens = num_entries;
    v->tokens = calloc(num_entries, sizeof(char *));
    v->lengths = calloc(num_entries, sizeof(int));

    for (uint32_t i = 0; i < num_entries; i++) {
        uint16_t byte_len;
        fread(&byte_len, 2, 1, f);
        if (byte_len > 0) {
            v->tokens[i] = malloc(byte_len + 1);
            fread(v->tokens[i], 1, byte_len, f);
            v->tokens[i][byte_len] = '\0';
            // Decode GPT-2 BPE byte encoding (Ġ→space, Ċ→newline, etc.)
            v->lengths[i] = bpe_decode_inplace(v->tokens[i], byte_len);
        }
    }

    fclose(f);
    printf("[vocab] Loaded %d tokens\n", num_entries);
    return v;
}

const char *decode_token(Vocabulary *v, int token_id) {
    if (token_id < 0 || token_id >= v->num_tokens || !v->tokens[token_id]) {
        return "<unk>";
    }
    return v->tokens[token_id];
}

// ============================================================================
// Prompt tokens loader
// ============================================================================

// PromptTokens defined in infer_api.h

static PromptTokens *load_prompt_tokens(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    PromptTokens *pt = calloc(1, sizeof(PromptTokens));
    fread(&pt->count, 4, 1, f);
    pt->ids = malloc(pt->count * sizeof(uint32_t));
    fread(pt->ids, 4, pt->count, f);
    fclose(f);
    return pt;
}

// ============================================================================
// C BPE tokenizer (replaces Python encode_prompt.py)
// ============================================================================
#define TOKENIZER_IMPL
#include "tokenizer.h"

static bpe_tokenizer g_tokenizer;
static int g_tokenizer_loaded = 0;

static void init_tokenizer(void) {
    if (g_tokenizer_loaded) return;
    // Try model_path first, then cwd, then metal_infer/
    char model_tok[1024];
    const char *paths[4];
    int n = 0;
    if (cfg.model_path[0]) {
        snprintf(model_tok, sizeof(model_tok), "%s/tokenizer.bin", cfg.model_path);
        paths[n++] = model_tok;
    }
    paths[n++] = "tokenizer.bin";
    paths[n++] = "metal_infer/tokenizer.bin";
    paths[n] = NULL;
    for (int i = 0; paths[i]; i++) {
        if (access(paths[i], R_OK) == 0) {
            if (bpe_load(&g_tokenizer, paths[i]) == 0) {
                g_tokenizer_loaded = 1;
                return;
            }
        }
    }
    // Try model directory (iOS: tokenizer downloaded with model)
    if (cfg.model_path[0]) {
        char model_tok[1024];
        snprintf(model_tok, sizeof(model_tok), "%s/tokenizer.bin", cfg.model_path);
        if (access(model_tok, R_OK) == 0) {
            if (bpe_load(&g_tokenizer, model_tok) == 0) {
                g_tokenizer_loaded = 1;
                return;
            }
        }
    }
    // Try app bundle (iOS)
    @autoreleasepool {
        NSString *bundlePath = [[NSBundle mainBundle] pathForResource:@"tokenizer" ofType:@"bin"];
        if (bundlePath) {
            if (bpe_load(&g_tokenizer, [bundlePath UTF8String]) == 0) {
                g_tokenizer_loaded = 1;
                return;
            }
        }
    }
    fprintf(stderr, "WARNING: tokenizer.bin not found, tokenization will fail\n");
}

PromptTokens *encode_prompt_text_to_tokens(const char *text) {
    init_tokenizer();
    if (!g_tokenizer_loaded) return NULL;

    // Allocate output buffer (generous: 4 tokens per character worst case)
    int max_ids = (int)strlen(text) * 4 + 256;
    uint32_t *ids = malloc(max_ids * sizeof(uint32_t));
    if (!ids) return NULL;

    int n = bpe_encode(&g_tokenizer, text, ids, max_ids);
    if (n < 0) { free(ids); return NULL; }

    PromptTokens *pt = calloc(1, sizeof(PromptTokens));
    pt->ids = ids;
    pt->count = n;

    fprintf(stderr, "Tokens (%d): [", n);
    for (int i = 0; i < n && i < 20; i++) {
        if (i > 0) fprintf(stderr, ", ");
        fprintf(stderr, "%u", ids[i]);
    }
    if (n > 20) fprintf(stderr, ", ...");
    fprintf(stderr, "]\n");

    return pt;
}

// ============================================================================
// CPU computation kernels
// ============================================================================

// 4-bit dequant matvec: out[out_dim] = W * x[in_dim]
// W is stored as packed uint32 (8 x 4-bit values per uint32)
// scales/biases are bfloat16 per group
static void cpu_dequant_matvec(
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    const float *x, float *out,
    int out_dim, int in_dim, int group_size
) {
    int num_groups = in_dim / group_size;
    int packed_per_group = group_size / 8;
    int packed_cols = in_dim / 8;

    for (int row = 0; row < out_dim; row++) {
        float acc = 0.0f;
        const uint32_t *w_row = W + row * packed_cols;
        const uint16_t *s_row = scales + row * num_groups;
        const uint16_t *b_row = biases + row * num_groups;

        for (int g = 0; g < num_groups; g++) {
            float scale = bf16_to_f32(s_row[g]);
            float bias = bf16_to_f32(b_row[g]);
            int base_packed = g * packed_per_group;
            int base_x = g * group_size;

            for (int p = 0; p < packed_per_group; p++) {
                uint32_t packed = w_row[base_packed + p];
                int x_base = base_x + p * 8;

                for (int n = 0; n < 8; n++) {
                    uint32_t nibble = (packed >> (n * 4)) & 0xF;
                    acc += ((float)nibble * scale + bias) * x[x_base + n];
                }
            }
        }
        out[row] = acc;
    }
}

// N-bit dequant matvec: out[out_dim] = W * x[in_dim]
// bits = 4 or 8. W is packed uint32 (8/bits values per uint32).
static void cpu_dequant_matvec_nbits(
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    const float *x, float *out,
    int out_dim, int in_dim, int group_size, int bits
) {
    if (bits == 4) {
        cpu_dequant_matvec(W, scales, biases, x, out, out_dim, in_dim, group_size);
        return;
    }
    // 8-bit: 4 values per uint32
    int vals_per_u32 = 32 / bits;
    int mask = (1 << bits) - 1;
    int num_groups = in_dim / group_size;
    int packed_per_group = group_size / vals_per_u32;
    int packed_cols = in_dim / vals_per_u32;

    for (int row = 0; row < out_dim; row++) {
        float acc = 0.0f;
        const uint32_t *w_row = W + row * packed_cols;
        const uint16_t *s_row = scales + row * num_groups;
        const uint16_t *b_row = biases + row * num_groups;

        for (int g = 0; g < num_groups; g++) {
            float scale = bf16_to_f32(s_row[g]);
            float bias = bf16_to_f32(b_row[g]);
            int base_packed = g * packed_per_group;
            int base_x = g * group_size;

            for (int p = 0; p < packed_per_group; p++) {
                uint32_t packed = w_row[base_packed + p];
                int x_base = base_x + p * vals_per_u32;
                for (int n = 0; n < vals_per_u32; n++) {
                    uint32_t val = (packed >> (n * bits)) & mask;
                    acc += ((float)val * scale + bias) * x[x_base + n];
                }
            }
        }
        out[row] = acc;
    }
}

// RMS normalization: out = x * w / rms(x)
void cpu_rms_norm(const float *x, const uint16_t *w_bf16, float *out, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = sqrtf(sum_sq / dim + eps);
    float inv_rms = 1.0f / rms;
    for (int i = 0; i < dim; i++) {
        float weight = bf16_to_f32(w_bf16[i]);
        out[i] = x[i] * inv_rms * weight;
    }
}

// SwiGLU: out = silu(gate) * up
static void cpu_swiglu(const float *gate, const float *up, float *out, int dim) {
    for (int i = 0; i < dim; i++) {
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        out[i] = silu_g * up[i];
    }
}

// Sigmoid
static float cpu_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Softmax over a vector
static void cpu_softmax(float *x, int dim) {
    float max_val = x[0];
    for (int i = 1; i < dim; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < dim; i++) {
        x[i] *= inv_sum;
    }
}

// Top-K: find K largest indices from scores[dim]
static void cpu_topk(const float *scores, int dim, int K, int *indices, float *values) {
    // Simple selection sort for small K
    // Initialize with -inf
    for (int k = 0; k < K; k++) {
        values[k] = -1e30f;
        indices[k] = 0;
    }

    for (int i = 0; i < dim; i++) {
        // Check if this score beats the smallest in our top-K
        int min_k = 0;
        for (int k = 1; k < K; k++) {
            if (values[k] < values[min_k]) min_k = k;
        }
        if (scores[i] > values[min_k]) {
            values[min_k] = scores[i];
            indices[min_k] = i;
        }
    }
}

// Normalize top-K weights to sum to 1
static void cpu_normalize_weights(float *weights, int K) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) sum += weights[k];
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int k = 0; k < K; k++) weights[k] *= inv;
    }
}

// Unified expert routing: handles both softmax (Qwen) and sigmoid (MiniMax).
// gate_scores[num_experts] is the raw router output (MODIFIED in place).
// routing_bias may be NULL (Qwen) or [num_experts] correction bias (MiniMax).
// On return: expert_indices[K] and expert_weights[K] are populated and normalized.
static void cpu_route_experts(float *gate_scores, int num_experts, int K,
                              const float *routing_bias,
                              int *expert_indices, float *expert_weights) {
    if (cfg.scoring_func == 1) {
        // Sigmoid routing (MiniMax): sigmoid → bias-corrected top-K → normalize original scores
        for (int i = 0; i < num_experts; i++)
            gate_scores[i] = 1.0f / (1.0f + expf(-gate_scores[i]));

        if (routing_bias) {
            // Top-K on bias-corrected scores, but extract original sigmoid weights
            float biased[num_experts];
            for (int i = 0; i < num_experts; i++)
                biased[i] = gate_scores[i] + routing_bias[i];
            cpu_topk(biased, num_experts, K, expert_indices, expert_weights);
            // Replace biased weights with original sigmoid scores
            for (int k = 0; k < K; k++)
                expert_weights[k] = gate_scores[expert_indices[k]];
        } else {
            cpu_topk(gate_scores, num_experts, K, expert_indices, expert_weights);
        }
        // Normalize: scores / (sum + eps)
        float sum = 0.0f;
        for (int k = 0; k < K; k++) sum += expert_weights[k];
        float inv = 1.0f / (sum + 1e-20f);
        for (int k = 0; k < K; k++) expert_weights[k] *= inv;
    } else {
        // Softmax routing (Qwen): softmax → top-K → normalize
        cpu_softmax(gate_scores, num_experts);
        cpu_topk(gate_scores, num_experts, K, expert_indices, expert_weights);
        cpu_normalize_weights(expert_weights, K);
    }
}

// Element-wise add: dst += src
__attribute__((unused))
static void cpu_vec_add(float *dst, const float *src, int dim) {
    for (int i = 0; i < dim; i++) dst[i] += src[i];
}

// Element-wise multiply-add: dst += scale * src
static void cpu_vec_madd(float *dst, const float *src, float scale, int dim) {
    for (int i = 0; i < dim; i++) dst[i] += scale * src[i];
}

// Element-wise multiply: dst = a * b
__attribute__((unused))
static void cpu_vec_mul(float *dst, const float *a, const float *b, int dim) {
    for (int i = 0; i < dim; i++) dst[i] = a[i] * b[i];
}

// Copy
static void cpu_vec_copy(float *dst, const float *src, int dim) {
    memcpy(dst, src, dim * sizeof(float));
}

// Zero
__attribute__((unused))
static void cpu_vec_zero(float *dst, int dim) {
    memset(dst, 0, dim * sizeof(float));
}

// Argmax
int cpu_argmax(const float *x, int dim) {
    int best = 0;
    float best_val = x[0];
    for (int i = 1; i < dim; i++) {
        if (x[i] > best_val) {
            best_val = x[i];
            best = i;
        }
    }
    return best;
}

// SiLU activation
static void cpu_silu(float *x, int dim) {
    for (int i = 0; i < dim; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

// Conv1d depthwise: one step (for incremental inference)
// Input: conv_state[kernel_size-1][channels] + new_input[channels]
// Output: result[channels]
// Weight: [channels, kernel_size, 1] stored as bf16
// This is a depthwise conv1d: each channel is independent
static void cpu_conv1d_step(
    const float *conv_state,    // [(kernel_size-1) * channels] row-major
    const float *new_input,     // [channels]
    const uint16_t *weight_bf16, // [channels * kernel_size] flattened
    float *out,                 // [channels]
    int channels,
    int kernel_size
) {
    // For each channel, compute dot product of [conv_state..., new_input] with weight
    for (int c = 0; c < channels; c++) {
        float acc = 0.0f;
        // Process previous states from conv_state
        for (int k = 0; k < kernel_size - 1; k++) {
            float w = bf16_to_f32(weight_bf16[c * kernel_size + k]);
            acc += conv_state[k * channels + c] * w;
        }
        // Process new input (last position in kernel)
        float w = bf16_to_f32(weight_bf16[c * kernel_size + (kernel_size - 1)]);
        acc += new_input[c] * w;
        out[c] = acc;
    }
    // Apply SiLU
    cpu_silu(out, channels);
}

// ============================================================================
// Metal context for GPU-accelerated matmuls
// ============================================================================

// Maximum number of batched matmul output slots.
// Used for encoding multiple matmuls into one command buffer.
#define MAX_BATCH_SLOTS 8

typedef struct {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLLibrary>              library;
    id<MTLComputePipelineState> matvec_v3;
    id<MTLComputePipelineState> matvec_v5;  // LUT dequant variant
    id<MTLComputePipelineState> matvec_fast;  // for in_dim > 4096
    id<MTLComputePipelineState> matvec_2bit;  // 2-bit expert dequant kernel
    id<MTLComputePipelineState> matvec_iq3_xxs; // GGUF IQ3_XXS streamed expert kernel
    id<MTLComputePipelineState> matvec_iq4_xs; // GGUF IQ4_XS streamed expert kernel
    id<MTLComputePipelineState> matvec_q5_k; // GGUF Q5_K streamed expert kernel
    id<MTLComputePipelineState> matvec_q8_0;  // GGUF Q8_0 resident tensor path
    id<MTLComputePipelineState> matvec_q6_k;  // GGUF Q6_K resident tensor path
    // NAX (Metal 4 / M5+) pipelines
    id<MTLLibrary>              nax_library;   // compiled with Metal 4.0 (NULL if not available)
    id<MTLComputePipelineState> nax_dequant;   // 4-bit → half dequant kernel
    id<MTLComputePipelineState> nax_f32_to_half; // float32 → half conversion
    id<MTLComputePipelineState> nax_gemm;      // NAX tensor matmul2d
    id<MTLComputePipelineState> nax_extract;   // extract row 0 from column-major output
    id<MTLComputePipelineState> nax_transpose; // column-major → row-major transpose
    id<MTLBuffer>               nax_w_half;    // dequantized weight buffer (half)
    id<MTLBuffer>               nax_x_half;    // input converted to half (padded to 32 rows)
    id<MTLBuffer>               nax_c_buf;     // padded output buffer [32, VOCAB_SIZE]
    // Per-projection NAX weight caches for batched prefill (dequantized to half, cached)
    id<MTLBuffer>               nax_pfb_w_cache[8]; // cached dequantized weights per projection slot
    int                         nax_pfb_w_valid[8]; // 1 if the cached weights are valid for this layer
    int                         nax_pfb_cached_layer; // layer index whose weights are cached (-1 = none)
    int                         has_nax;       // 1 if NAX hardware available
    // Batched prefill GEMM
    id<MTLComputePipelineState> gemm_batch;    // dequant_gemm_4bit_batch kernel
    id<MTLBuffer>               buf_pfb_input; // [MAX_PFB * HIDDEN_DIM floats] batched input
    id<MTLBuffer>               buf_pfb_out[8];// [MAX_PFB * max_proj_dim floats] per projection slot

    id<MTLComputePipelineState> rms_norm_sum;
    id<MTLComputePipelineState> rms_norm_apply;
    id<MTLComputePipelineState> rms_norm_apply_bf16;
    id<MTLComputePipelineState> residual_add;
    id<MTLComputePipelineState> swiglu;
    id<MTLComputePipelineState> fused_gate_up;  // fused gate+up+SwiGLU kernel (4-bit only)
    // FP16 accumulation variants (experimental — toggled via g_use_fp16_accum)
    id<MTLComputePipelineState> matvec_v3_fp16;
    id<MTLComputePipelineState> matvec_2bit_fp16;
    id<MTLComputePipelineState> fused_gate_up_fp16;
    // Prefill kernels
    id<MTLComputePipelineState> prefill_causal_attn;
    id<MTLComputePipelineState> prefill_rms_norm;
    id<MTLComputePipelineState> prefill_residual_norm;
    id<MTLComputePipelineState> prefill_swiglu;
    id<MTLComputePipelineState> prefill_combine;
    id<MTLComputePipelineState> prefill_q_rope_norm;
    id<MTLComputePipelineState> prefill_kv_cache;
    // GPU attention pipelines
    id<MTLComputePipelineState> attn_scores_pipe;
    id<MTLComputePipelineState> attn_softmax_pipe;
    id<MTLComputePipelineState> attn_values_pipe;
    id<MTLComputePipelineState> sigmoid_gate_pipe;
    // FP8 E4M3 KV cache attention pipelines (opt-in via g_use_fp8_kv)
    id<MTLComputePipelineState> attn_scores_fp8_pipe;
    id<MTLComputePipelineState> attn_values_fp8_pipe;
    // Fused online softmax attention (replaces 3-kernel pipeline)
    id<MTLComputePipelineState> fused_attention_pipe;
    id<MTLComputePipelineState> fused_attention_fp8_pipe;
    // Function-constant specialized fused attention
    id<MTLComputePipelineState> fused_attention_fc_pipe;
    // Reusable buffers for attention matmuls
    id<MTLBuffer> buf_input;     // input vector [cfg.hidden_dim or max projection input]
    id<MTLBuffer> buf_output;    // output vector [max projection output]
    id<MTLBuffer> wf_buf;        // the mmap'd weight file as a Metal buffer (first chunk or sole)
    // Split weight file into multiple Metal buffers for files > 4GB (Metal limit on iOS)
    #define MAX_WF_CHUNKS 4
    id<MTLBuffer> wf_chunks[MAX_WF_CHUNKS];
    size_t wf_chunk_offsets[MAX_WF_CHUNKS]; // byte offset of each chunk from mmap base
    size_t wf_chunk_sizes[MAX_WF_CHUNKS];
    int wf_num_chunks;
    void *wf_mmap_base;         // base of mmap'd weight file
    size_t wf_mmap_size;        // size of mmap'd weight file
    // iOS staging mode: when wf_num_chunks == 0, GPU dispatches copy tensor data
    // into this reusable staging buffer instead of using zero-copy Metal buffers.
    // This avoids Metal tracking the full 5.5GB weight file as GPU memory.
    id<MTLBuffer> wf_staging;   // reusable staging buffer (~50MB, iOS only)
    size_t wf_staging_used;     // bytes currently packed into staging buffer
    // Batched matmul output slots (preallocated, reused across dispatches)
    id<MTLBuffer> batch_out[MAX_BATCH_SLOTS];
    // Reusable buffers for expert computation (avoids per-expert alloc)
    // Legacy single-expert buffers (kept for gpu_expert_forward compat)
    id<MTLBuffer> buf_expert_data;   // holds one expert's packed weights (cfg.expert_size_4bit bytes)
    id<MTLBuffer> buf_expert_input;  // h_post input [cfg.hidden_dim floats]
    id<MTLBuffer> buf_expert_gate;   // gate_proj output [cfg.moe_intermediate floats]
    id<MTLBuffer> buf_expert_up;     // up_proj output [cfg.moe_intermediate floats]
    id<MTLBuffer> buf_expert_act;    // SwiGLU output [cfg.moe_intermediate floats]
    id<MTLBuffer> buf_expert_out;    // down_proj output [cfg.hidden_dim floats]
    // Multi-expert buffers: K independent sets so all experts can be encoded
    // into a SINGLE command buffer (no per-expert commit+wait).
    // Each expert k uses slot [k].
    // Double-buffered: set A (data) for GPU compute, set B (data_B) for background pread.
    // Gate/up/act/out only need one set (GPU uses them after pread completes).
    #define MAX_K 16
    id<MTLBuffer> buf_multi_expert_data[MAX_K];   // [cfg.expert_size_4bit bytes] each — buffer set A
    id<MTLBuffer> buf_multi_expert_data_B[MAX_K]; // [cfg.expert_size_4bit bytes] each — buffer set B (prefetch)
    id<MTLBuffer> buf_multi_expert_gate[MAX_K];   // [cfg.moe_intermediate floats]
    id<MTLBuffer> buf_multi_expert_up[MAX_K];     // [cfg.moe_intermediate floats]
    id<MTLBuffer> buf_multi_expert_act[MAX_K];    // [cfg.moe_intermediate floats]
    id<MTLBuffer> buf_multi_expert_out[MAX_K];    // [cfg.hidden_dim floats]
    id<MTLBuffer> buf_multi_expert_input;         // [cfg.hidden_dim floats] (shared, read-only during dispatch)
    // Shared expert buffers for fused CMD2 (shared gate/up computed in CMD1,
    // SwiGLU + down_proj in CMD2 alongside routed experts)
    id<MTLBuffer> buf_shared_gate;   // [cfg.shared_intermediate floats]
    id<MTLBuffer> buf_shared_up;     // [cfg.shared_intermediate floats]
    id<MTLBuffer> buf_shared_act;    // [cfg.shared_intermediate floats] (SwiGLU output)
    id<MTLBuffer> buf_shared_out;    // [cfg.hidden_dim floats] (down_proj output)
    // Fused o_proj+norm+routing buffers (eliminates 1 cmd buffer per layer)
    id<MTLBuffer> buf_residual;     // [cfg.hidden_dim floats] holds residual for GPU add
    id<MTLBuffer> buf_h_mid;        // [cfg.hidden_dim floats] residual+oproj result
    id<MTLBuffer> buf_sum_sq;       // [1 float] for RMS norm reduction
    // GPU attention buffers (for full attention layers)
    id<MTLBuffer> __strong *buf_kv_k;  // K cache per full-attn layer (float or uchar when FP8)
    id<MTLBuffer> __strong *buf_kv_v;  // V cache per full-attn layer (float or uchar when FP8)
    // FP8 per-position scale buffers (1 float per cached position per layer)
    id<MTLBuffer> __strong *buf_kv_k_scales;  // [gpu_kv floats] per full-attn layer (FP8 only)
    id<MTLBuffer> __strong *buf_kv_v_scales;  // [gpu_kv floats] per full-attn layer (FP8 only)
    id<MTLBuffer> buf_attn_q;       // [cfg.num_attn_heads * cfg.head_dim floats] all query heads
    id<MTLBuffer> buf_attn_scores;  // [cfg.num_attn_heads * cfg.max_seq_len floats] all heads' scores
    id<MTLBuffer> buf_attn_out;     // [cfg.num_attn_heads * cfg.head_dim floats] full attention output
    id<MTLBuffer> buf_attn_gate;    // [cfg.num_attn_heads * cfg.head_dim floats] sigmoid gate
    // CMD3 GPU-side combine buffers (weighted_sum + residual + norm on GPU)
    id<MTLComputePipelineState> moe_combine_residual;  // fused combine kernel
    id<MTLBuffer> buf_moe_hidden;     // [cfg.hidden_dim floats] GPU combine output (hidden state)
    id<MTLBuffer> buf_combine_params; // [10 floats] expert weights[8] + shared_gate_score + padding
    id<MTLBuffer> buf_cmd3_sum_sq;    // [1 float] for RMS norm reduction in CMD3
    // Shared event for CPU-GPU synchronization (async pipeline)
    id<MTLSharedEvent> pipeline_event;   // CPU signals when buf_input is ready
    uint64_t event_value;                // monotonically increasing event counter
    // GPU delta-net (gated_delta_net_step) and conv1d pipelines
    id<MTLComputePipelineState> delta_net_step;  // gated_delta_net_step kernel
    id<MTLComputePipelineState> delta_net_step_fused;  // pass 2+3 merged (saves ~1M reads/token)
    id<MTLComputePipelineState> delta_net_step_batched;  // prefill chunked gated-delta kernel
    id<MTLComputePipelineState> conv1d_step;     // conv1d_step kernel
    id<MTLComputePipelineState> conv1d_step_batched;     // prefill chunked conv1d kernel
    id<MTLComputePipelineState> rms_norm_qk;     // per-head RMS normalize for q and k
    id<MTLComputePipelineState> rms_norm_qk_batched;     // batched per-head RMS normalize
    id<MTLComputePipelineState> compute_decay_beta; // g_decay and beta_gate for delta-net
    id<MTLComputePipelineState> compute_decay_beta_batched; // batched g_decay and beta_gate
    id<MTLComputePipelineState> gated_rms_norm;  // z-gated output normalization
    id<MTLComputePipelineState> gated_rms_norm_batched;  // batched z-gated output normalization
    // Persistent GPU state buffers for linear attention layers
    id<MTLBuffer> __strong *buf_delta_state;   // [v_heads*v_dim*k_dim] float per layer
    id<MTLBuffer> __strong *buf_conv_state;     // [(kernel-1)*conv_dim] float per layer
    // Scratch buffers for delta-net inputs/outputs
    id<MTLBuffer> buf_delta_q;        // [cfg.linear_total_key=2048] float
    id<MTLBuffer> buf_delta_k;        // [cfg.linear_total_key=2048] float
    id<MTLBuffer> buf_delta_v;        // [cfg.linear_total_value=4096] float
    id<MTLBuffer> buf_delta_g_decay;  // [cfg.linear_num_v_heads=32] float
    id<MTLBuffer> buf_delta_beta;     // [cfg.linear_num_v_heads=32] float
    id<MTLBuffer> buf_delta_output;   // [cfg.linear_total_value=4096] float
    id<MTLBuffer> buf_conv_input;     // [cfg.linear_conv_dim=8192] float
    id<MTLBuffer> buf_conv_output;    // [cfg.linear_conv_dim=8192] float
} MetalCtx;

static MetalCtx *g_metal = NULL;

MetalCtx *metal_setup(void) {
    // Set GPU KV cache size from model config (avoids over-allocating on iOS)
    if (cfg.max_seq_len > 0 && cfg.max_seq_len < GPU_KV_SEQ) {
        GPU_KV_SEQ = cfg.max_seq_len;
    }
    printf("[metal] GPU_KV_SEQ = %d\n", GPU_KV_SEQ);

    MetalCtx *ctx = calloc(1, sizeof(MetalCtx));
    // Allocate dynamic buffer arrays based on config
    ctx->buf_kv_k       = (__strong id<MTLBuffer> *)calloc(cfg.num_full_attn_layers, sizeof(id<MTLBuffer>));
    ctx->buf_kv_v       = (__strong id<MTLBuffer> *)calloc(cfg.num_full_attn_layers, sizeof(id<MTLBuffer>));
    ctx->buf_kv_k_scales = (__strong id<MTLBuffer> *)calloc(cfg.num_full_attn_layers, sizeof(id<MTLBuffer>));
    ctx->buf_kv_v_scales = (__strong id<MTLBuffer> *)calloc(cfg.num_full_attn_layers, sizeof(id<MTLBuffer>));
    ctx->buf_delta_state = (__strong id<MTLBuffer> *)calloc(cfg.num_linear_layers, sizeof(id<MTLBuffer>));
    ctx->buf_conv_state  = (__strong id<MTLBuffer> *)calloc(cfg.num_linear_layers, sizeof(id<MTLBuffer>));
    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) {
        fprintf(stderr, "ERROR: No Metal device\n");
        free(ctx); return NULL;
    }
    printf("[metal] Device: %s\n", [[ctx->device name] UTF8String]);

    ctx->queue = [ctx->device newCommandQueue];
    if (!ctx->queue) {
        fprintf(stderr, "ERROR: No command queue\n");
        free(ctx); return NULL;
    }

    // Load Metal shaders
    NSError *error = nil;
    double t0 = now_ms();

    // Try pre-compiled default.metallib first (iOS app bundle, or macOS with embedded metallib)
    ctx->library = [ctx->device newDefaultLibrary];
    if (ctx->library) {
        printf("[metal] Loaded pre-compiled Metal library: %.0f ms\n", now_ms() - t0);
    } else {
        // Fallback: compile shaders from source at runtime (macOS CLI)
        NSArray *paths = @[@"shaders.metal", @"metal_infer/shaders.metal"];
        NSString *src = nil;
        for (NSString *p in paths) {
            src = [NSString stringWithContentsOfFile:p encoding:NSUTF8StringEncoding error:&error];
            if (src) break;
        }
        if (!src) {
            fprintf(stderr, "ERROR: Cannot find shaders.metal\n");
            free(ctx); return NULL;
        }

        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        opts.mathMode = MTLMathModeFast;
        opts.languageVersion = MTLLanguageVersion3_1;
        ctx->library = [ctx->device newLibraryWithSource:src options:opts error:&error];
        if (!ctx->library) {
            fprintf(stderr, "ERROR: Shader compile failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            free(ctx); return NULL;
        }
        printf("[metal] Shader compile: %.0f ms\n", now_ms() - t0);
    }

    // Create pipelines
    id<MTLComputePipelineState> (^makePipe)(NSString *) = ^(NSString *name) {
        id<MTLFunction> fn = [ctx->library newFunctionWithName:name];
        if (!fn) { fprintf(stderr, "ERROR: shader '%s' not found\n", [name UTF8String]); return (id<MTLComputePipelineState>)nil; }
        NSError *e2 = nil;
        id<MTLComputePipelineState> ps = [ctx->device newComputePipelineStateWithFunction:fn error:&e2];
        if (!ps) { fprintf(stderr, "ERROR: pipeline '%s': %s\n", [name UTF8String], [[e2 localizedDescription] UTF8String]); }
        return ps;
    };

    ctx->gemm_batch    = makePipe(@"dequant_gemm_4bit_batch");
    ctx->prefill_causal_attn = makePipe(@"prefill_causal_attn");
    ctx->prefill_rms_norm    = makePipe(@"prefill_rms_norm_bf16");
    ctx->prefill_residual_norm = makePipe(@"prefill_residual_norm_bf16");
    ctx->prefill_swiglu      = makePipe(@"prefill_swiglu");
    ctx->prefill_combine     = makePipe(@"prefill_combine");
    ctx->prefill_q_rope_norm = makePipe(@"prefill_q_rope_norm_bf16");
    ctx->prefill_kv_cache    = makePipe(@"prefill_kv_cache_bf16");
    ctx->matvec_v3     = makePipe(@"dequant_matvec_4bit_v3");
    ctx->matvec_v5     = makePipe(@"dequant_matvec_4bit_v5");  // LUT variant (no uint→float conversions)
    ctx->matvec_fast   = makePipe(@"dequant_matvec_4bit_fast");
    ctx->matvec_2bit   = makePipe(@"dequant_matvec_2bit");
    ctx->rms_norm_sum  = makePipe(@"rms_norm_sum_sq");
    ctx->rms_norm_apply = makePipe(@"rms_norm_apply");
    ctx->rms_norm_apply_bf16 = makePipe(@"rms_norm_apply_bf16");
    ctx->residual_add  = makePipe(@"residual_add");
    ctx->swiglu        = makePipe(@"swiglu_fused");
    ctx->fused_gate_up = makePipe(@"fused_gate_up_swiglu");
    // FP16 accumulation variants (optional — float path is fallback)
    ctx->matvec_v3_fp16    = makePipe(@"dequant_matvec_4bit_v3_fp16");
    ctx->matvec_2bit_fp16  = makePipe(@"dequant_matvec_2bit_fp16");
    ctx->fused_gate_up_fp16 = makePipe(@"fused_gate_up_swiglu_fp16");
    if (!ctx->matvec_v3_fp16)    fprintf(stderr, "[metal] WARNING: fp16 matvec_v3 pipeline failed (float fallback)\n");
    if (!ctx->matvec_2bit_fp16)  fprintf(stderr, "[metal] WARNING: fp16 matvec_2bit pipeline failed (float fallback)\n");
    if (!ctx->fused_gate_up_fp16) fprintf(stderr, "[metal] WARNING: fp16 fused_gate_up pipeline failed (float fallback)\n");
    ctx->attn_scores_pipe  = makePipe(@"attn_scores_batched");
    ctx->attn_softmax_pipe = makePipe(@"attn_softmax_batched");
    ctx->attn_values_pipe  = makePipe(@"attn_values_batched");
    ctx->sigmoid_gate_pipe = makePipe(@"sigmoid_gate");
    // FP8 E4M3 KV cache attention kernels
    ctx->attn_scores_fp8_pipe = makePipe(@"attn_scores_fp8");
    ctx->attn_values_fp8_pipe = makePipe(@"attn_values_fp8");
    // Fused online softmax attention (single kernel replaces 3-kernel pipeline)
    ctx->fused_attention_pipe     = makePipe(@"fused_attention_online");
    ctx->fused_attention_fp8_pipe = makePipe(@"fused_attention_online_fp8");
    if (!ctx->fused_attention_pipe)     fprintf(stderr, "[metal] WARNING: fused_attention_online pipeline failed (3-kernel fallback)\n");
    if (!ctx->fused_attention_fp8_pipe) fprintf(stderr, "[metal] WARNING: fused_attention_online_fp8 pipeline failed (3-kernel fallback)\n");
    // Function-constant specialized fused attention
    {
        MTLFunctionConstantValues *fcv = [[MTLFunctionConstantValues alloc] init];
        bool use_fp8 = (g_use_fp8_kv != 0);
        [fcv setConstantValue:&use_fp8 type:MTLDataTypeBool atIndex:0];
        NSError *fc_err = nil;
        id<MTLFunction> fc_fn = [ctx->library newFunctionWithName:@"fused_attention_online_fc"
                                                   constantValues:fcv
                                                            error:&fc_err];
        if (fc_fn) {
            ctx->fused_attention_fc_pipe = [ctx->device newComputePipelineStateWithFunction:fc_fn error:&fc_err];
            if (ctx->fused_attention_fc_pipe) {
                printf("[metal] Function-constant fused attention pipeline ready (FP8=%d)\n", use_fp8);
            } else {
                fprintf(stderr, "[metal] WARNING: fused_attention_fc pipeline creation failed: %s\n",
                        [[fc_err localizedDescription] UTF8String]);
            }
        } else {
            fprintf(stderr, "[metal] WARNING: fused_attention_online_fc function not found: %s\n",
                    fc_err ? [[fc_err localizedDescription] UTF8String] : "unknown");
        }
    }
    ctx->moe_combine_residual = makePipe(@"moe_combine_residual");
    ctx->delta_net_step    = makePipe(@"gated_delta_net_step");
    ctx->delta_net_step_fused = makePipe(@"gated_delta_net_step_fused");
    ctx->delta_net_step_batched = makePipe(@"gated_delta_net_step_batched");
    ctx->conv1d_step       = makePipe(@"conv1d_step");
    ctx->conv1d_step_batched = makePipe(@"conv1d_step_batched");
    ctx->rms_norm_qk       = makePipe(@"rms_norm_qk");
    ctx->rms_norm_qk_batched = makePipe(@"rms_norm_qk_batched");
    ctx->compute_decay_beta = makePipe(@"compute_decay_beta");
    ctx->compute_decay_beta_batched = makePipe(@"compute_decay_beta_batched");
    ctx->gated_rms_norm    = makePipe(@"gated_rms_norm");
    ctx->gated_rms_norm_batched = makePipe(@"gated_rms_norm_batched");
    if (!ctx->moe_combine_residual) fprintf(stderr, "[metal] WARNING: moe_combine_residual pipeline failed\n");
    if (!ctx->delta_net_step_fused) fprintf(stderr, "[metal] WARNING: gated_delta_net_step_fused pipeline failed (using unfused fallback)\n");
    if (!ctx->prefill_q_rope_norm) fprintf(stderr, "[metal] WARNING: prefill_q_rope_norm_bf16 pipeline failed (prefill fallback)\n");
    if (!ctx->prefill_kv_cache) fprintf(stderr, "[metal] WARNING: prefill_kv_cache_bf16 pipeline failed (prefill fallback)\n");
    if (!ctx->delta_net_step) fprintf(stderr, "[metal] WARNING: gated_delta_net_step pipeline failed (CPU fallback)\n");
    if (!ctx->delta_net_step_batched) fprintf(stderr, "[metal] WARNING: gated_delta_net_step_batched pipeline failed (prefill fallback)\n");
    if (!ctx->conv1d_step)    fprintf(stderr, "[metal] WARNING: conv1d_step pipeline failed (CPU fallback)\n");
    if (!ctx->conv1d_step_batched) fprintf(stderr, "[metal] WARNING: conv1d_step_batched pipeline failed (prefill fallback)\n");
    if (!ctx->rms_norm_qk)       fprintf(stderr, "[metal] WARNING: rms_norm_qk pipeline failed (CPU fallback)\n");
    if (!ctx->rms_norm_qk_batched) fprintf(stderr, "[metal] WARNING: rms_norm_qk_batched pipeline failed (prefill fallback)\n");
    if (!ctx->compute_decay_beta) fprintf(stderr, "[metal] WARNING: compute_decay_beta pipeline failed (CPU fallback)\n");
    if (!ctx->compute_decay_beta_batched) fprintf(stderr, "[metal] WARNING: compute_decay_beta_batched pipeline failed (prefill fallback)\n");
    if (!ctx->gated_rms_norm)     fprintf(stderr, "[metal] WARNING: gated_rms_norm pipeline failed (CPU fallback)\n");
    if (!ctx->gated_rms_norm_batched) fprintf(stderr, "[metal] WARNING: gated_rms_norm_batched pipeline failed (prefill fallback)\n");

    // ---- NAX (Metal 4 / M5+) ----
    ctx->has_nax = 0;
    if (@available(macOS 26.2, *)) {
        if ([ctx->device supportsFamily:(MTLGPUFamily)5002]) {
            // Try compiling NAX shaders with Metal 4.0
            NSArray *nax_paths = @[@"nax_gemm.metal", @"metal_infer/nax_gemm.metal"];
            NSString *nax_src = nil;
            for (NSString *p in nax_paths) {
                nax_src = [NSString stringWithContentsOfFile:p encoding:NSUTF8StringEncoding error:&error];
                if (nax_src) break;
            }
            if (nax_src) {
                MTLCompileOptions *nax_opts = [[MTLCompileOptions alloc] init];
                nax_opts.languageVersion = (MTLLanguageVersion)0x40000;  // Metal 4.0
                nax_opts.mathMode = MTLMathModeFast;
                ctx->nax_library = [ctx->device newLibraryWithSource:nax_src options:nax_opts error:&error];
                if (ctx->nax_library) {
                    id<MTLComputePipelineState> (^naxPipe)(NSString *) = ^(NSString *name) {
                        id<MTLFunction> fn = [ctx->nax_library newFunctionWithName:name];
                        if (!fn) return (id<MTLComputePipelineState>)nil;
                        NSError *e2 = nil;
                        return [ctx->device newComputePipelineStateWithFunction:fn error:&e2];
                    };
                    ctx->nax_dequant = naxPipe(@"nax_dequant_4bit");
                    ctx->nax_f32_to_half = naxPipe(@"nax_f32_to_half");
                    ctx->nax_gemm = naxPipe(@"nax_gemm_f32_input");
                    ctx->nax_extract = naxPipe(@"nax_extract_row0");
                    ctx->nax_transpose = naxPipe(@"nax_transpose_cm_to_rm");
                    if (ctx->nax_dequant && ctx->nax_gemm && ctx->nax_f32_to_half) {
                        ctx->has_nax = 1;
                        // Pre-allocate buffers for LM head (largest projection: vocab_size × hidden_dim)
                        // M is padded to 32 for NAX tile alignment
                        ctx->nax_w_half = [ctx->device newBufferWithLength:(size_t)cfg.vocab_size * cfg.hidden_dim * sizeof(uint16_t)
                                                                   options:MTLResourceStorageModeShared];
                        ctx->nax_x_half = [ctx->device newBufferWithLength:(size_t)32 * cfg.hidden_dim * sizeof(uint16_t)
                                                                   options:MTLResourceStorageModeShared];
                        ctx->nax_c_buf = [ctx->device newBufferWithLength:(size_t)32 * cfg.vocab_size * sizeof(float)
                                                                   options:MTLResourceStorageModeShared];
                        // Initialize batched prefill NAX weight cache state
                        ctx->nax_pfb_cached_layer = -1;
                        memset(ctx->nax_pfb_w_valid, 0, sizeof(ctx->nax_pfb_w_valid));
                        memset(ctx->nax_pfb_w_cache, 0, sizeof(ctx->nax_pfb_w_cache));
                        printf("[metal] NAX (Metal 4) enabled — tensor matmul for LM head + batched prefill\n");
                    }
                }
            }
            if (!ctx->has_nax) {
                printf("[metal] NAX shader compile failed, using standard kernels\n");
            }
        }
    }

    if (!ctx->matvec_v3 || !ctx->matvec_fast) {
        fprintf(stderr, "ERROR: Required Metal pipeline missing\n");
        free(ctx); return NULL;
    }

    // Allocate reusable buffers (large enough for biggest projection)
    // Q proj output is 16384 floats, lm_head output is 248320 floats
    // o_proj input is 8192, linear attn out_proj input is 8192
    size_t max_out = cfg.vocab_size * sizeof(float);  // lm_head is largest
    size_t max_in = cfg.linear_total_value * sizeof(float);  // 8192 floats (linear_attn out_proj)
    if (max_in < (size_t)(cfg.num_attn_heads * cfg.head_dim) * sizeof(float)) {
        max_in = (size_t)(cfg.num_attn_heads * cfg.head_dim) * sizeof(float);  // o_proj input = 8192
    }
    ctx->buf_input  = [ctx->device newBufferWithLength:max_in  options:MTLResourceStorageModeShared];
    ctx->buf_output = [ctx->device newBufferWithLength:max_out options:MTLResourceStorageModeShared];

    // Batched matmul output slots — each large enough for the biggest projection
    // q_proj = 16384 floats, qkv_proj = 12288, z_proj = 8192, o_proj = 4096
    // lm_head (248320) uses buf_output directly, not batched.
    {
        size_t slot_size = (size_t)(cfg.num_attn_heads * cfg.head_dim * 2) * sizeof(float);  // 16384 floats
        if (slot_size < (size_t)cfg.linear_conv_dim * sizeof(float))
            slot_size = (size_t)cfg.linear_conv_dim * sizeof(float);  // 12288 floats
        for (int i = 0; i < MAX_BATCH_SLOTS; i++) {
            ctx->batch_out[i] = [ctx->device newBufferWithLength:slot_size
                                                         options:MTLResourceStorageModeShared];
        }
    }

    // Expert computation buffers (reused across all experts and layers)
    ctx->buf_expert_data  = [ctx->device newBufferWithLength:cfg.expert_size_4bit
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_input = [ctx->device newBufferWithLength:cfg.hidden_dim * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_gate  = [ctx->device newBufferWithLength:cfg.moe_intermediate * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_up    = [ctx->device newBufferWithLength:cfg.moe_intermediate * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_act   = [ctx->device newBufferWithLength:cfg.moe_intermediate * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_out   = [ctx->device newBufferWithLength:cfg.hidden_dim * sizeof(float)
                                                     options:MTLResourceStorageModeShared];

    // Multi-expert buffers: K independent slots (double-buffered data)
    // Expert data buffers use 2MB-aligned backing memory for DMA efficiency.
    // The pread DMA controller transfers 3.6x faster with 2MB alignment vs 16KB.
    ctx->buf_multi_expert_input = [ctx->device newBufferWithLength:cfg.hidden_dim * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
    size_t expert_alloc_size = (cfg.expert_size_4bit + 2*1024*1024 - 1) & ~(2*1024*1024 - 1);  // round up to 2MB
    for (int k = 0; k < MAX_K; k++) {
        // 2MB-aligned allocation for optimal DMA throughput
        void *aligned_data = NULL, *aligned_data_b = NULL;
        posix_memalign(&aligned_data,   2*1024*1024, expert_alloc_size);
        posix_memalign(&aligned_data_b, 2*1024*1024, expert_alloc_size);
        memset(aligned_data, 0, expert_alloc_size);
        memset(aligned_data_b, 0, expert_alloc_size);
        ctx->buf_multi_expert_data[k] = [ctx->device newBufferWithBytesNoCopy:aligned_data
                                                                       length:expert_alloc_size
                                                                      options:MTLResourceStorageModeShared
                                                                  deallocator:nil];
        ctx->buf_multi_expert_data_B[k] = [ctx->device newBufferWithBytesNoCopy:aligned_data_b
                                                                         length:expert_alloc_size
                                                                        options:MTLResourceStorageModeShared
                                                                    deallocator:nil];
        ctx->buf_multi_expert_gate[k] = [ctx->device newBufferWithLength:cfg.moe_intermediate * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        ctx->buf_multi_expert_up[k]   = [ctx->device newBufferWithLength:cfg.moe_intermediate * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        ctx->buf_multi_expert_act[k]  = [ctx->device newBufferWithLength:cfg.moe_intermediate * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        ctx->buf_multi_expert_out[k]  = [ctx->device newBufferWithLength:cfg.hidden_dim * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
    }

    // Shared expert buffers (for fused CMD2)
    ctx->buf_shared_gate = [ctx->device newBufferWithLength:cfg.shared_intermediate * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    ctx->buf_shared_up   = [ctx->device newBufferWithLength:cfg.shared_intermediate * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    ctx->buf_shared_act  = [ctx->device newBufferWithLength:cfg.shared_intermediate * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    ctx->buf_shared_out  = [ctx->device newBufferWithLength:cfg.hidden_dim * sizeof(float)
                                                    options:MTLResourceStorageModeShared];

    // Fused o_proj+norm+routing buffers
    ctx->buf_residual = [ctx->device newBufferWithLength:cfg.hidden_dim * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    ctx->buf_h_mid    = [ctx->device newBufferWithLength:cfg.hidden_dim * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    ctx->buf_sum_sq   = [ctx->device newBufferWithLength:sizeof(float)
                                                 options:MTLResourceStorageModeShared];

    // CMD3 GPU-side combine buffers
    ctx->buf_moe_hidden    = [ctx->device newBufferWithLength:cfg.hidden_dim * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
    ctx->buf_combine_params = [ctx->device newBufferWithLength:10 * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
    ctx->buf_cmd3_sum_sq    = [ctx->device newBufferWithLength:sizeof(float)
                                                        options:MTLResourceStorageModeShared];

    // GPU attention buffers
    {
        size_t kv_dim = cfg.num_kv_heads * cfg.head_dim;  // 512
        size_t kv_elem_size = g_use_fp8_kv ? sizeof(uint8_t) : sizeof(float);
        size_t kv_cache_size = GPU_KV_SEQ * kv_dim * kv_elem_size;
        for (int i = 0; i < cfg.num_full_attn_layers; i++) {
            ctx->buf_kv_k[i] = [ctx->device newBufferWithLength:kv_cache_size
                                                        options:MTLResourceStorageModeShared];
            ctx->buf_kv_v[i] = [ctx->device newBufferWithLength:kv_cache_size
                                                        options:MTLResourceStorageModeShared];
            if (g_use_fp8_kv) {
                ctx->buf_kv_k_scales[i] = [ctx->device newBufferWithLength:GPU_KV_SEQ * sizeof(float)
                                                                   options:MTLResourceStorageModeShared];
                ctx->buf_kv_v_scales[i] = [ctx->device newBufferWithLength:GPU_KV_SEQ * sizeof(float)
                                                                   options:MTLResourceStorageModeShared];
            }
        }
        if (g_use_fp8_kv) {
            printf("[metal] FP8 E4M3 KV cache enabled — 4x memory reduction (%.1f MB per layer)\n",
                   kv_cache_size / 1e6);
        }
        ctx->buf_attn_q      = [ctx->device newBufferWithLength:cfg.num_attn_heads * cfg.head_dim * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        ctx->buf_attn_scores = [ctx->device newBufferWithLength:(size_t)cfg.num_attn_heads * GPU_KV_SEQ * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        ctx->buf_attn_out    = [ctx->device newBufferWithLength:cfg.num_attn_heads * cfg.head_dim * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        ctx->buf_attn_gate   = [ctx->device newBufferWithLength:cfg.num_attn_heads * cfg.head_dim * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        printf("[metal] GPU attention buffers: %d KV caches (%.1f MB each), scores buf %.1f MB\n",
               cfg.num_full_attn_layers, kv_cache_size / 1e6,
               (double)(cfg.num_attn_heads * cfg.max_seq_len * sizeof(float)) / 1e6);
    }

    // Persistent GPU state buffers for delta-net (linear attention layers)
    if (ctx->delta_net_step) {
        for (int i = 0; i < cfg.num_linear_layers; i++) {
            ctx->buf_delta_state[i] = [ctx->device newBufferWithLength:(size_t)cfg.linear_num_v_heads*cfg.linear_value_dim*cfg.linear_key_dim*sizeof(float)
                                                               options:MTLResourceStorageModeShared];
            memset([ctx->buf_delta_state[i] contents], 0, (size_t)cfg.linear_num_v_heads*cfg.linear_value_dim*cfg.linear_key_dim*sizeof(float));
            ctx->buf_conv_state[i] = [ctx->device newBufferWithLength:(cfg.conv_kernel_size-1)*(size_t)cfg.linear_conv_dim*sizeof(float)
                                                              options:MTLResourceStorageModeShared];
            memset([ctx->buf_conv_state[i] contents], 0, (cfg.conv_kernel_size-1)*(size_t)cfg.linear_conv_dim*sizeof(float));
        }
        // Scratch buffers for delta-net inputs/outputs (allocated once, reused)
        ctx->buf_delta_q       = [ctx->device newBufferWithLength:cfg.linear_total_key*sizeof(float)    options:MTLResourceStorageModeShared];
        ctx->buf_delta_k       = [ctx->device newBufferWithLength:cfg.linear_total_key*sizeof(float)    options:MTLResourceStorageModeShared];
        ctx->buf_delta_v       = [ctx->device newBufferWithLength:cfg.linear_total_value*sizeof(float)  options:MTLResourceStorageModeShared];
        ctx->buf_delta_g_decay = [ctx->device newBufferWithLength:cfg.linear_num_v_heads*sizeof(float)  options:MTLResourceStorageModeShared];
        ctx->buf_delta_beta    = [ctx->device newBufferWithLength:cfg.linear_num_v_heads*sizeof(float)  options:MTLResourceStorageModeShared];
        ctx->buf_delta_output  = [ctx->device newBufferWithLength:cfg.linear_total_value*sizeof(float)  options:MTLResourceStorageModeShared];
        ctx->buf_conv_input    = [ctx->device newBufferWithLength:cfg.linear_conv_dim*sizeof(float)     options:MTLResourceStorageModeShared];
        ctx->buf_conv_output   = [ctx->device newBufferWithLength:cfg.linear_conv_dim*sizeof(float)     options:MTLResourceStorageModeShared];
        size_t state_bytes = (size_t)cfg.linear_num_v_heads*cfg.linear_value_dim*cfg.linear_key_dim*sizeof(float);
        size_t conv_bytes = (cfg.conv_kernel_size-1)*(size_t)cfg.linear_conv_dim*sizeof(float);
        printf("[metal] Delta-net GPU buffers: %d layers (%.1f MB state + %.1f MB scratch)\n",
               cfg.num_linear_layers,
               cfg.num_linear_layers * (state_bytes + conv_bytes) / 1e6,
               (cfg.linear_total_key*2+cfg.linear_total_value*2+cfg.linear_num_v_heads*2+cfg.linear_conv_dim*2) * sizeof(float) / 1e6);
    }

    // Create shared event for CPU-GPU async pipeline
    ctx->pipeline_event = [ctx->device newSharedEvent];
    ctx->event_value = 0;

    // ---- Prefill batch buffers ----
    if (g_prefill_batch > 1 && ctx->gemm_batch) {
        size_t pfb = g_prefill_batch;
        // Input buffer: N × hidden_dim floats
        ctx->buf_pfb_input = [ctx->device newBufferWithLength:pfb * MAX_HIDDEN_DIM * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        // Output slots for projections (largest = QKV at 12288 or Q at 16384)
        size_t max_proj = 16384;  // NUM_ATTN_HEADS * HEAD_DIM * 2 for full attn Q
        for (int i = 0; i < 8; i++) {
            ctx->buf_pfb_out[i] = [ctx->device newBufferWithLength:pfb * max_proj * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
        }
        printf("[metal] Prefill batch buffers allocated (pfb=%d, %.1f MB)\n",
               (int)pfb, (pfb * MAX_HIDDEN_DIM + pfb * max_proj * 8) * sizeof(float) / 1e6);
        if (g_prefill_batch > 1) {
            printf("[metal] Prefill config: batch=%d, skip_experts=%d, batched_linear=%d\n",
                   g_prefill_batch, effective_prefill_skip_experts(), !g_disable_batched_linear);
        }
        if (g_prefill_batch > MAX_PFB_GPU) {
            printf("[metal] Prefill GPU chunk size capped at %d tokens per dispatch\n", MAX_PFB_GPU);
        }
    }

    printf("[metal] Inference pipelines ready (multi-expert[%d] + shared buffers allocated)\n", MAX_K);
    return ctx;
}

// Reset delta-net and conv GPU state buffers (call at start of new generation)
void reset_delta_net_state(void) {
    if (!g_metal || !g_metal->delta_net_step) return;
    for (int i = 0; i < cfg.num_linear_layers; i++) {
        if (g_metal->buf_delta_state[i])
            memset([g_metal->buf_delta_state[i] contents], 0, (size_t)cfg.linear_num_v_heads*cfg.linear_value_dim*cfg.linear_key_dim*sizeof(float));
        if (g_metal->buf_conv_state[i])
            memset([g_metal->buf_conv_state[i] contents], 0, (cfg.conv_kernel_size-1)*(size_t)cfg.linear_conv_dim*sizeof(float));
    }
}

// Wrap the mmap'd weight file as Metal buffer(s) (zero-copy on unified memory).
// Metal enforces a hard 4GB per-buffer limit. Files >4GB get two overlapping buffers.
// mmap returns page-aligned addresses, Metal requires the same.
// On Apple Silicon, page size is 16KB.
#define METAL_MAX_BUF ((size_t)4096 * 1024 * 1024 - 16384)  // 4GB - 1 page
#define WF_STAGING_SIZE ((size_t)50 * 1024 * 1024)  // legacy, kept for compile compat
void metal_set_weights(MetalCtx *ctx, void *data, size_t size) {
    size_t page_size = 16384;
    ctx->wf_mmap_base = data;
    ctx->wf_mmap_size = size;
    ctx->wf_num_chunks = 0;
    ctx->wf_staging = nil;
    ctx->wf_staging_used = 0;
    ctx->wf_buf = nil;

    size_t aligned_size = (size + page_size - 1) & ~(page_size - 1);

    if (size <= METAL_MAX_BUF) {
        // Fits in one buffer — zero-copy wrap
        ctx->wf_buf = [ctx->device newBufferWithBytesNoCopy:data
                                                     length:aligned_size
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
        if (ctx->wf_buf) {
            ctx->wf_chunks[0] = ctx->wf_buf;
            ctx->wf_chunk_offsets[0] = 0;
            ctx->wf_chunk_sizes[0] = aligned_size;
            ctx->wf_num_chunks = 1;
            printf("[metal] Weight file wrapped as single Metal buffer (%.2f GB)\n", size / 1e9);
        } else {
            fprintf(stderr, "WARNING: Cannot wrap weight file (%.2f GB) — CPU fallback\n", size / 1e9);
        }
    } else {
        // >4GB: too large for Metal buffers on memory-constrained devices.
        // CPU fallback for non-expert matmuls. The mmap is still used — just no Metal wrapper.
        // All guards (g_metal->wf_buf, wf_num_chunks > 0) will route to CPU path.
        printf("[metal] Weight file %.2f GB exceeds 4GB Metal buffer limit.\n", size / 1e9);
        printf("[metal]   wf_buf=%s wf_num_chunks=%d — CPU fallback for non-expert matmuls.\n",
               ctx->wf_buf ? "SET" : "nil", ctx->wf_num_chunks);
    }
    printf("[metal] metal_set_weights done: wf_buf=%s wf_num_chunks=%d\n",
           ctx->wf_buf ? "SET" : "nil", ctx->wf_num_chunks);
}

// GPU dequant matvec: out[out_dim] = W_4bit * x[in_dim]
// W_packed, scales, biases are pointers into mmap'd weight file
// x_f32 is CPU float array, result written back to out_f32
//
// We wrap the ENTIRE mmap'd weight file as a single Metal buffer and use
// byte offsets to point each shader argument at the right tensor.
// This avoids per-tensor buffer creation and the page-alignment constraint.

// ============================================================================
// Weight buffer resolution: chunk mode (macOS) vs staging mode (iOS)
// ============================================================================

// Reset staging buffer offset — call at start of each command buffer encode
static inline void metal_staging_reset(MetalCtx *ctx) {
    ctx->wf_staging_used = 0;
}

// Stage a tensor into the staging buffer. Returns buffer + offset within staging.
// Tensors are packed sequentially; call metal_staging_reset() between command buffers.
static inline void metal_stage(MetalCtx *ctx, const void *ptr, size_t size,
                                id<MTLBuffer> *out_buf, NSUInteger *out_offset) {
    if (ctx->wf_staging_used + size > WF_STAGING_SIZE) {
        fprintf(stderr, "ERROR: staging buffer overflow: used=%zu + size=%zu > %zu\n",
                ctx->wf_staging_used, size, (size_t)WF_STAGING_SIZE);
        // Reset and overwrite from beginning (data corruption but won't crash)
        ctx->wf_staging_used = 0;
    }
    memcpy((char *)[ctx->wf_staging contents] + ctx->wf_staging_used, ptr, size);
    *out_buf = ctx->wf_staging;
    *out_offset = (NSUInteger)ctx->wf_staging_used;
    ctx->wf_staging_used += size;
    // Align to 16 bytes for Metal buffer offset requirements
    ctx->wf_staging_used = (ctx->wf_staging_used + 15) & ~(size_t)15;
}

// Find which Metal buffer chunk contains a given pointer, return the buffer and offset within it.
// In staging mode (wf_num_chunks == 0, iOS), copies tensor data into staging buffer.
// The `size` parameter is only used in staging mode — pass 0 on macOS chunk path.
static inline void metal_find_chunk_sized(MetalCtx *ctx, const void *ptr, size_t size,
                                           id<MTLBuffer> *out_buf, NSUInteger *out_offset) {
    // No Metal weight buffers (>4GB on iOS) — should not be called
    if (ctx->wf_num_chunks == 0 && !ctx->wf_staging) {
        fprintf(stderr, "BUG: metal_find_chunk called with no weight buffers! Using buf_input as dummy.\n");
        *out_buf = ctx->buf_input;
        *out_offset = 0;
        return;
    }
    // Staging mode (iOS): copy tensor into staging buffer
    if (ctx->wf_staging && ctx->wf_num_chunks == 0) {
        metal_stage(ctx, ptr, size, out_buf, out_offset);
        return;
    }
    // Chunk mode: find the zero-copy Metal buffer containing this pointer
    size_t abs_off = (const char *)ptr - (const char *)ctx->wf_mmap_base;
    for (int i = ctx->wf_num_chunks - 1; i >= 0; i--) {
        if (abs_off >= ctx->wf_chunk_offsets[i]) {
            NSUInteger local_off = (NSUInteger)(abs_off - ctx->wf_chunk_offsets[i]);
            if (local_off < [ctx->wf_chunks[i] length]) {
                *out_buf = ctx->wf_chunks[i];
                *out_offset = local_off;
                return;
            }
        }
    }
    fprintf(stderr, "ERROR: metal_find_chunk: ptr offset %zu (%.2f GB) not in any chunk!\n",
            abs_off, abs_off / 1e9);
    for (int i = 0; i < ctx->wf_num_chunks; i++) {
        fprintf(stderr, "  chunk %d: offset=%zu size=%zu\n",
                i, ctx->wf_chunk_offsets[i], ctx->wf_chunk_sizes[i]);
    }
    *out_buf = ctx->wf_chunks[0];
    *out_offset = (NSUInteger)abs_off;
}

// Legacy wrapper — used by sites that don't know tensor size (macOS only, chunk mode)
static inline void metal_find_chunk(MetalCtx *ctx, const void *ptr,
                                     id<MTLBuffer> *out_buf, NSUInteger *out_offset) {
    metal_find_chunk_sized(ctx, ptr, 0, out_buf, out_offset);
}

// Convenience macros for macOS chunk mode (no size needed)
#define WF_OFF(ctx, ptr) ({ \
    id<MTLBuffer> _b; NSUInteger _o; \
    metal_find_chunk((ctx), (ptr), &_b, &_o); _o; })
#define WF_BUF(ctx, ptr) ({ \
    id<MTLBuffer> _b; NSUInteger _o; \
    metal_find_chunk((ctx), (ptr), &_b, &_o); _b; })

static void gpu_dequant_matvec(
    MetalCtx *ctx,
    const void *W_packed, const void *scales, const void *biases,
    const float *x_f32, float *out_f32,
    uint32_t out_dim, uint32_t in_dim, uint32_t group_size
) {
    // Copy input to Metal buffer
    memcpy([ctx->buf_input contents], x_f32, in_dim * sizeof(float));

    size_t o_size = (size_t)out_dim * sizeof(float);

    // Find correct Metal buffer chunk and offset for each tensor
    id<MTLBuffer> w_buf, s_buf, b_buf;
    NSUInteger w_off, s_off, b_off;
    size_t w_size = (size_t)out_dim * in_dim / 8;  // 4-bit packed
    size_t num_groups = (in_dim + group_size - 1) / group_size;
    size_t sb_size = (size_t)out_dim * num_groups * sizeof(uint16_t);
    metal_staging_reset(ctx);
    metal_find_chunk_sized(ctx, W_packed, w_size, &w_buf, &w_off);
    metal_find_chunk_sized(ctx, scales, sb_size, &s_buf, &s_off);
    metal_find_chunk_sized(ctx, biases, sb_size, &b_buf, &b_off);

    // Ensure output buffer is large enough
    id<MTLBuffer> o_buf = ctx->buf_output;
    if (o_size > [o_buf length]) {
        o_buf = [ctx->device newBufferWithLength:o_size options:MTLResourceStorageModeShared];
    }

    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];

    // v3 shader uses x_shared[4096], so can only handle in_dim <= 4096
    // For larger in_dim (e.g. o_proj with in_dim=8192), use matvec_fast
    int use_v3 = (in_dim <= 4096);
    {
        id<MTLComputePipelineState> pipe;
        if (g_use_fp16_accum && use_v3 && ctx->matvec_v3_fp16) {
            pipe = ctx->matvec_v3_fp16;
        } else {
            pipe = use_v3 ? ctx->matvec_v3 : ctx->matvec_fast;
        }
        [enc setComputePipelineState:pipe];
    }
    [enc setBuffer:w_buf        offset:w_off atIndex:0];
    [enc setBuffer:s_buf        offset:s_off atIndex:1];
    [enc setBuffer:b_buf        offset:b_off atIndex:2];
    [enc setBuffer:ctx->buf_input offset:0   atIndex:3];
    [enc setBuffer:o_buf        offset:0     atIndex:4];
    [enc setBytes:&out_dim      length:4     atIndex:5];
    [enc setBytes:&in_dim       length:4     atIndex:6];
    [enc setBytes:&group_size   length:4     atIndex:7];

    if (use_v3) {
        // v3: tiled threadgroups, 256 threads, 8 rows per TG
        uint32_t num_tgs = (out_dim + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        // fast: one threadgroup per output row, 64 threads per TG
        NSUInteger tg_size = 64;
        [enc dispatchThreadgroups:MTLSizeMake(out_dim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    }
    [enc endEncoding];
    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    // Copy result back
    memcpy(out_f32, [o_buf contents], o_size);
}

// Track whether LM head weights have been dequantized to nax_w_half
static int g_nax_lmhead_dequantized = 0;

// NAX 2-pass dispatch: dequant weights to half (cached), convert input to half, then NAX GEMM
// Only used for large projections (LM head) where NAX provides significant speedup
static void gpu_nax_dequant_gemm(
    MetalCtx *ctx,
    const void *W_packed, const void *scales, const void *biases,
    const float *x_f32, float *out_f32,
    uint32_t out_dim, uint32_t in_dim, uint32_t group_size
) {
    uint32_t num_groups = in_dim / group_size;
    NSUInteger w_off = (NSUInteger)((const char *)W_packed - (const char *)[ctx->wf_buf contents]);
    NSUInteger s_off = (NSUInteger)((const char *)scales   - (const char *)[ctx->wf_buf contents]);
    NSUInteger b_off = (NSUInteger)((const char *)biases   - (const char *)[ctx->wf_buf contents]);

    // Pad M to NAX tile size (32) to avoid cooperative tensor store overflow
    const uint32_t NAX_BM = 32;
    uint32_t M_padded = NAX_BM;  // always pad to at least one full tile

    // Ensure NAX buffers are large enough
    size_t w_half_size = (size_t)out_dim * in_dim * sizeof(uint16_t);
    if ([ctx->nax_w_half length] < w_half_size) {
        ctx->nax_w_half = [ctx->device newBufferWithLength:w_half_size options:MTLResourceStorageModeShared];
    }
    size_t x_half_size = (size_t)M_padded * in_dim * sizeof(uint16_t);
    if ([ctx->nax_x_half length] < x_half_size) {
        ctx->nax_x_half = [ctx->device newBufferWithLength:x_half_size options:MTLResourceStorageModeShared];
    }

    // Copy input to buf_input (only 1 row of actual data, zero-pad rest for M_padded=32)
    size_t input_padded_size = (size_t)M_padded * in_dim * sizeof(float);
    if ([ctx->buf_input length] < input_padded_size) {
        ctx->buf_input = [ctx->device newBufferWithLength:input_padded_size options:MTLResourceStorageModeShared];
    }
    memcpy([ctx->buf_input contents], x_f32, in_dim * sizeof(float));
    memset((char *)[ctx->buf_input contents] + in_dim * sizeof(float), 0,
           (M_padded - 1) * in_dim * sizeof(float));

    // Pass 1: Dequantize 4-bit weights → half (cached after first call)
    if (!g_nax_lmhead_dequantized) {
        id<MTLCommandBuffer> deq_cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [deq_cmd computeCommandEncoder];
        [enc setComputePipelineState:ctx->nax_dequant];
        [enc setBuffer:ctx->wf_buf    offset:w_off atIndex:0];
        [enc setBuffer:ctx->wf_buf    offset:s_off atIndex:1];
        [enc setBuffer:ctx->wf_buf    offset:b_off atIndex:2];
        [enc setBuffer:ctx->nax_w_half offset:0    atIndex:3];
        [enc setBytes:&out_dim length:4 atIndex:4];
        [enc setBytes:&in_dim  length:4 atIndex:5];
        uint32_t total_groups = out_dim * num_groups;
        uint32_t tg_size = 256;
        uint32_t num_tgs = (total_groups + tg_size - 1) / tg_size;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [deq_cmd commit]; [deq_cmd waitUntilCompleted];
        g_nax_lmhead_dequantized = 1;
        fprintf(stderr, "[nax] LM head weights dequantized to half (cached, %.1f MB)\n",
                (double)out_dim * in_dim * 2 / 1e6);
    }

    // NAX GEMM: C[M_padded, N] = x_f32[M_padded, K] @ W_half[N, K]^T
    // nax_gemm_f32_input converts f32→half inline (no separate pass)
    id<MTLBuffer> c_buf = ctx->nax_c_buf;

    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    [enc setComputePipelineState:ctx->nax_gemm];
    [enc setBuffer:ctx->buf_input  offset:0 atIndex:0];  // A[M_padded, K] float32
    [enc setBuffer:ctx->nax_w_half offset:0 atIndex:1];  // B[N, K] half (cached)
    [enc setBuffer:c_buf           offset:0 atIndex:2];  // C[M_padded, N] float32
    [enc setBytes:&M_padded length:4 atIndex:3];
    [enc setBytes:&out_dim  length:4 atIndex:4];
    [enc setBytes:&in_dim   length:4 atIndex:5];
    int gx = (out_dim + 31) / 32;
    int gy = (M_padded + 31) / 32;
    [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];

    // Extract row 0 from column-major output on GPU (avoids slow strided CPU readback)
    {
        id<MTLComputeCommandEncoder> enc2 = [cmdbuf computeCommandEncoder];
        [enc2 setComputePipelineState:ctx->nax_extract];
        [enc2 setBuffer:c_buf          offset:0 atIndex:0];  // column-major input
        [enc2 setBuffer:ctx->buf_output offset:0 atIndex:1]; // row-major output
        [enc2 setBytes:&out_dim  length:4 atIndex:2];
        [enc2 setBytes:&M_padded length:4 atIndex:3];
        [enc2 dispatchThreads:MTLSizeMake(out_dim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(out_dim, (uint32_t)1024), 1, 1)];
        [enc2 endEncoding];
    }

    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    memcpy(out_f32, [ctx->buf_output contents], out_dim * sizeof(float));
}

// Wrapper: use GPU if available and weight buffer is set, CPU otherwise
static void fast_dequant_matvec(
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    const float *x, float *out,
    int out_dim, int in_dim, int group_size
) {
    int use_gpu = g_metal && (g_metal->wf_num_chunks > 0 || g_metal->wf_staging);
    if (use_gpu) {
        // Use NAX for LM head only (248K × hidden) — largest projection
        // WARNING: NAX with M=1 (single-token decode) is SLOWER than FMA due to 32×32 tile
        // padding overhead (31/32 of tile wasted). Only beneficial for batched decode (M≥4).
        // The --nax flag enables this path for benchmarking; not recommended for production M=1.
        if (g_metal->has_nax && !g_nax_disabled && out_dim > 100000) {
            gpu_nax_dequant_gemm(g_metal, W, scales, biases, x, out,
                                 (uint32_t)out_dim, (uint32_t)in_dim, (uint32_t)group_size);
        } else {
            // In staging mode, check if tensors fit in staging buffer.
            // LM head (~500MB) won't fit — fall back to CPU for that.
            size_t w_size = (size_t)out_dim * in_dim / 8;
            size_t num_groups = ((unsigned)in_dim + group_size - 1) / group_size;
            size_t sb_size = (size_t)out_dim * num_groups * sizeof(uint16_t);
            size_t total = w_size + sb_size + sb_size;
            if (g_metal->wf_staging && g_metal->wf_num_chunks == 0 && total > WF_STAGING_SIZE) {
                cpu_dequant_matvec(W, scales, biases, x, out, out_dim, in_dim, group_size);
                return;
            }
            gpu_dequant_matvec(g_metal, W, scales, biases, x, out,
                               (uint32_t)out_dim, (uint32_t)in_dim, (uint32_t)group_size);
        }
    } else {
        cpu_dequant_matvec(W, scales, biases, x, out, out_dim, in_dim, group_size);
    }
}

// ============================================================================
// Batched GPU matmul: encode N independent matmuls sharing the same input
// into ONE command buffer, reducing dispatch overhead by N-1 round-trips.
// ============================================================================

typedef struct {
    const void *W;           // packed weights (pointer into mmap'd file)
    const void *scales;      // scales (pointer into mmap'd file)
    const void *biases;      // biases (pointer into mmap'd file)
    float *out_cpu;          // CPU output pointer (result copied here after GPU finishes)
    uint32_t out_dim;
    uint32_t in_dim;
    uint32_t group_size;
    int batch_slot;          // which batch_out[slot] to use for GPU output
} BatchMatvecSpec;

// Run N matmuls in a single command buffer. All share the same input vector.
// The input is copied once; all outputs go to preallocated batch_out slots.
static void gpu_batch_matvec(
    MetalCtx *ctx,
    const float *x_f32, uint32_t x_dim,  // shared input
    BatchMatvecSpec *specs, int num_specs
) {
    // Copy input once
    memcpy([ctx->buf_input contents], x_f32, x_dim * sizeof(float));

    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];

    // Reset staging ONCE before the loop — all specs pack into the same staging buffer.
    // Each spec's data must remain valid until the command buffer completes.
    metal_staging_reset(ctx);

    for (int i = 0; i < num_specs; i++) {
        BatchMatvecSpec *s = &specs[i];
        id<MTLBuffer> w_buf, s_buf, b_buf;
        NSUInteger w_off, s_off, b_off;
        size_t w_size = (size_t)s->out_dim * s->in_dim / 8;
        size_t num_groups = (s->in_dim + s->group_size - 1) / s->group_size;
        size_t sb_size = (size_t)s->out_dim * num_groups * sizeof(uint16_t);
        metal_find_chunk_sized(ctx, s->W, w_size, &w_buf, &w_off);
        metal_find_chunk_sized(ctx, s->scales, sb_size, &s_buf, &s_off);
        metal_find_chunk_sized(ctx, s->biases, sb_size, &b_buf, &b_off);

        id<MTLBuffer> o_buf = ctx->batch_out[s->batch_slot];

        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        int use_v3 = (s->in_dim <= 4096);
        {
            id<MTLComputePipelineState> pipe;
            if (g_use_fp16_accum && use_v3 && ctx->matvec_v3_fp16) {
                pipe = ctx->matvec_v3_fp16;
            } else {
                pipe = use_v3 ? ctx->matvec_v3 : ctx->matvec_fast;
            }
            [enc setComputePipelineState:pipe];
        }
        [enc setBuffer:w_buf        offset:w_off atIndex:0];
        [enc setBuffer:s_buf        offset:s_off atIndex:1];
        [enc setBuffer:b_buf        offset:b_off atIndex:2];
        [enc setBuffer:ctx->buf_input offset:0   atIndex:3];
        [enc setBuffer:o_buf        offset:0     atIndex:4];
        [enc setBytes:&s->out_dim   length:4     atIndex:5];
        [enc setBytes:&s->in_dim    length:4     atIndex:6];
        [enc setBytes:&s->group_size length:4    atIndex:7];

        if (use_v3) {
            uint32_t num_tgs = (s->out_dim + 7) / 8;
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            [enc dispatchThreadgroups:MTLSizeMake(s->out_dim, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
        [enc endEncoding];
    }

    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    // Copy results back to CPU
    for (int i = 0; i < num_specs; i++) {
        BatchMatvecSpec *s = &specs[i];
        memcpy(s->out_cpu, [ctx->batch_out[s->batch_slot] contents],
               s->out_dim * sizeof(float));
    }
}

// ============================================================================
// Encode-only variants: add dispatches to an EXISTING command buffer.
// These do NOT commit — the caller batches multiple encode calls into one
// command buffer and commits once, eliminating per-dispatch overhead.
// ============================================================================

// Encode N matmuls into cmdbuf. Input must already be in ctx->buf_input.
static void gpu_encode_batch_matvec(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    BatchMatvecSpec *specs, int num_specs
) {
    // Reset staging ONCE — all specs accumulate in the same staging buffer.
    // The caller's command buffer reads all staged data after commit.
    metal_staging_reset(ctx);

    for (int i = 0; i < num_specs; i++) {
        BatchMatvecSpec *s = &specs[i];
        id<MTLBuffer> w_buf, s_buf, b_buf;
        NSUInteger w_off, s_off, b_off;
        size_t w_size = (size_t)s->out_dim * s->in_dim / 8;
        size_t num_groups = (s->in_dim + s->group_size - 1) / s->group_size;
        size_t sb_size = (size_t)s->out_dim * num_groups * sizeof(uint16_t);
        metal_find_chunk_sized(ctx, s->W, w_size, &w_buf, &w_off);
        metal_find_chunk_sized(ctx, s->scales, sb_size, &s_buf, &s_off);
        metal_find_chunk_sized(ctx, s->biases, sb_size, &b_buf, &b_off);

        id<MTLBuffer> o_buf = ctx->batch_out[s->batch_slot];

        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        int use_v3 = (s->in_dim <= 4096);
        {
            id<MTLComputePipelineState> pipe;
            if (g_use_fp16_accum && use_v3 && ctx->matvec_v3_fp16) {
                pipe = ctx->matvec_v3_fp16;
            } else {
                pipe = use_v3 ? ctx->matvec_v3 : ctx->matvec_fast;
            }
            [enc setComputePipelineState:pipe];
        }
        [enc setBuffer:w_buf        offset:w_off atIndex:0];
        [enc setBuffer:s_buf        offset:s_off atIndex:1];
        [enc setBuffer:b_buf        offset:b_off atIndex:2];
        [enc setBuffer:ctx->buf_input offset:0   atIndex:3];
        [enc setBuffer:o_buf        offset:0     atIndex:4];
        [enc setBytes:&s->out_dim   length:4     atIndex:5];
        [enc setBytes:&s->in_dim    length:4     atIndex:6];
        [enc setBytes:&s->group_size length:4    atIndex:7];

        if (use_v3) {
            uint32_t num_tgs = (s->out_dim + 7) / 8;
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            [enc dispatchThreadgroups:MTLSizeMake(s->out_dim, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
        [enc endEncoding];
    }
}

// Encode a batched GEMM using FMA kernel: project N tokens through one 4-bit weight matrix.
// Input: in_buf [N * in_dim floats], Output: out_buf [N * out_dim floats]
// Weight is 4-bit packed in wf_buf.
static void gpu_encode_pfb_gemm_fma(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    uint32_t out_dim, uint32_t in_dim, uint32_t group_size,
    int batch_n,
    id<MTLBuffer> in_buf, NSUInteger in_offset,
    id<MTLBuffer> out_buf, NSUInteger out_offset
) {
    if (!ctx->gemm_batch || !ctx->wf_buf) return;
    NSUInteger w_off = (NSUInteger)((const char *)W      - (const char *)[ctx->wf_buf contents]);
    NSUInteger s_off = (NSUInteger)((const char *)scales  - (const char *)[ctx->wf_buf contents]);
    NSUInteger b_off = (NSUInteger)((const char *)biases  - (const char *)[ctx->wf_buf contents]);
    uint32_t bn = (uint32_t)batch_n;

    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    [enc setComputePipelineState:ctx->gemm_batch];
    [enc setBuffer:ctx->wf_buf offset:w_off     atIndex:0];
    [enc setBuffer:ctx->wf_buf offset:s_off     atIndex:1];
    [enc setBuffer:ctx->wf_buf offset:b_off     atIndex:2];
    [enc setBuffer:in_buf      offset:in_offset atIndex:3];
    [enc setBuffer:out_buf     offset:out_offset atIndex:4];
    [enc setBytes:&out_dim    length:4 atIndex:5];
    [enc setBytes:&in_dim     length:4 atIndex:6];
    [enc setBytes:&group_size length:4 atIndex:7];
    [enc setBytes:&bn         length:4 atIndex:8];
    uint32_t num_tgs = (out_dim + 7) / 8;
    [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
}

// NAX batched prefill GEMM: dequant weights → half, NAX GEMM M=batch_n, transpose output.
// Encodes into an existing command buffer (no commit/wait — caller manages that).
// Weight dequant runs every dispatch (not cached across projections, unlike LM head).
static void gpu_encode_pfb_nax_gemm_ex(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    uint32_t out_dim, uint32_t in_dim, uint32_t group_size,
    int batch_n,
    id<MTLBuffer> in_buf, NSUInteger in_offset,
    id<MTLBuffer> out_buf, NSUInteger out_offset
) {
    if (!ctx->has_nax || !ctx->nax_gemm || !ctx->nax_dequant || !ctx->nax_transpose || !ctx->wf_buf) return;

    NSUInteger w_off = (NSUInteger)((const char *)W      - (const char *)[ctx->wf_buf contents]);
    NSUInteger s_off = (NSUInteger)((const char *)scales  - (const char *)[ctx->wf_buf contents]);
    NSUInteger b_off = (NSUInteger)((const char *)biases  - (const char *)[ctx->wf_buf contents]);
    uint32_t num_groups = in_dim / group_size;

    // Pad M to NAX tile size (32)
    const uint32_t NAX_BM = 32;
    uint32_t M_padded = ((uint32_t)batch_n + NAX_BM - 1) & ~(NAX_BM - 1);
    if (M_padded < NAX_BM) M_padded = NAX_BM;

    // Ensure nax_w_half is large enough for this projection's weights
    size_t w_half_size = (size_t)out_dim * in_dim * sizeof(uint16_t);
    if ([ctx->nax_w_half length] < w_half_size) {
        ctx->nax_w_half = [ctx->device newBufferWithLength:w_half_size options:MTLResourceStorageModeShared];
    }

    // Ensure nax_c_buf is large enough for padded output [M_padded, out_dim]
    size_t c_size = (size_t)M_padded * out_dim * sizeof(float);
    if ([ctx->nax_c_buf length] < c_size) {
        ctx->nax_c_buf = [ctx->device newBufferWithLength:c_size options:MTLResourceStorageModeShared];
    }

    // Zero-pad input if M_padded > batch_n: we rely on in_buf having valid data for batch_n rows
    // and the NAX kernel's bounds checking handles out-of-range M indices (loads 0.0f).

    // Pass 1: Dequantize 4-bit weights → half
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->nax_dequant];
        [enc setBuffer:ctx->wf_buf    offset:w_off atIndex:0];
        [enc setBuffer:ctx->wf_buf    offset:s_off atIndex:1];
        [enc setBuffer:ctx->wf_buf    offset:b_off atIndex:2];
        [enc setBuffer:ctx->nax_w_half offset:0    atIndex:3];
        [enc setBytes:&out_dim length:4 atIndex:4];
        [enc setBytes:&in_dim  length:4 atIndex:5];
        uint32_t total_groups = out_dim * num_groups;
        uint32_t tg_size = 256;
        uint32_t num_tgs = (total_groups + tg_size - 1) / tg_size;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
    }

    // Pass 2: NAX GEMM — C[M_padded, out_dim] = X[M_padded, in_dim] @ W_half[out_dim, in_dim]^T
    // nax_gemm_f32_input converts f32→half inline in threadgroup memory
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->nax_gemm];
        [enc setBuffer:in_buf         offset:in_offset atIndex:0];  // A[M, K] float32
        [enc setBuffer:ctx->nax_w_half offset:0        atIndex:1];  // B[N, K] half
        [enc setBuffer:ctx->nax_c_buf  offset:0        atIndex:2];  // C[M_padded, N] float32 column-major
        [enc setBytes:&M_padded length:4 atIndex:3];
        [enc setBytes:&out_dim  length:4 atIndex:4];
        [enc setBytes:&in_dim   length:4 atIndex:5];
        int gx = (out_dim + 31) / 32;
        int gy = (M_padded + 31) / 32;
        [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];
    }

    // Pass 3: Transpose column-major → row-major into output buffer
    {
        uint32_t M_actual = (uint32_t)batch_n;
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->nax_transpose];
        [enc setBuffer:ctx->nax_c_buf offset:0          atIndex:0];  // column-major input
        [enc setBuffer:out_buf        offset:out_offset  atIndex:1];  // row-major output
        [enc setBytes:&M_actual length:4 atIndex:2];
        [enc setBytes:&out_dim  length:4 atIndex:3];
        [enc setBytes:&M_padded length:4 atIndex:4];
        // 2D dispatch: threads_x = N, threads_y = M
        MTLSize grid = MTLSizeMake(((out_dim + 15) / 16) * 16, ((M_actual + 15) / 16) * 16, 1);
        MTLSize tg = MTLSizeMake(16, 16, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }
}

// Auto-selecting wrapper: FMA for batched prefill projections (dequant-on-the-fly, single pass),
// NAX for LM head only (large out_dim, cached dequant). NAX loses on small projections due to
// 3-pass overhead (dequant→GEMM→transpose) vs FMA's fused single-pass.
static void gpu_encode_pfb_gemm_ex(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    uint32_t out_dim, uint32_t in_dim, uint32_t group_size,
    int batch_n,
    id<MTLBuffer> in_buf, NSUInteger in_offset,
    id<MTLBuffer> out_buf, NSUInteger out_offset
) {
    gpu_encode_pfb_gemm_fma(ctx, cmdbuf, W, scales, biases,
                             out_dim, in_dim, group_size, batch_n,
                             in_buf, in_offset, out_buf, out_offset);
}

// Convenience: uses buf_pfb_input as input, buf_pfb_out[slot] as output
static void gpu_encode_pfb_gemm(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    uint32_t out_dim, uint32_t in_dim, uint32_t group_size,
    int batch_n, int out_slot
) {
    gpu_encode_pfb_gemm_ex(ctx, cmdbuf, W, scales, biases,
                           out_dim, in_dim, group_size, batch_n,
                           ctx->buf_pfb_input, 0,
                           ctx->buf_pfb_out[out_slot], 0);
}

// Copy batch results from GPU buffers back to CPU pointers.
static void gpu_flush_batch_results(MetalCtx *ctx, BatchMatvecSpec *specs, int num_specs) {
    for (int i = 0; i < num_specs; i++) {
        BatchMatvecSpec *s = &specs[i];
        memcpy(s->out_cpu, [ctx->batch_out[s->batch_slot] contents],
               s->out_dim * sizeof(float));
    }
}

// Encode a single matvec reading from buf_expert_act into buf_expert_out,
// using weight pointers into the mmap'd weight file.
// Used for shared expert down_proj which reads from a different input than
// the attention projections.
static void gpu_encode_dequant_matvec_with_io_bufs(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    const void *W, const void *scales, const void *biases,
    id<MTLBuffer> in_buf, id<MTLBuffer> out_buf,
    uint32_t out_dim, uint32_t in_dim, uint32_t group_size
) {
    id<MTLBuffer> w_buf, s_buf, b_buf;
    NSUInteger w_off, s_off, b_off;
    size_t w_size = (size_t)out_dim * in_dim / 8;  // 4-bit packed
    size_t num_groups = (in_dim + group_size - 1) / group_size;
    size_t sb_size = (size_t)out_dim * num_groups * sizeof(uint16_t);
    metal_staging_reset(ctx);
    metal_find_chunk_sized(ctx, W, w_size, &w_buf, &w_off);
    metal_find_chunk_sized(ctx, scales, sb_size, &s_buf, &s_off);
    metal_find_chunk_sized(ctx, biases, sb_size, &b_buf, &b_off);

    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    int use_v3 = (in_dim <= 4096);
    {
        id<MTLComputePipelineState> pipe;
        if (g_use_fp16_accum && use_v3 && ctx->matvec_v3_fp16) {
            pipe = ctx->matvec_v3_fp16;
        } else {
            pipe = use_v3 ? ctx->matvec_v3 : ctx->matvec_fast;
        }
        [enc setComputePipelineState:pipe];
    }
    [enc setBuffer:w_buf offset:w_off atIndex:0];
    [enc setBuffer:s_buf offset:s_off atIndex:1];
    [enc setBuffer:b_buf offset:b_off atIndex:2];
    [enc setBuffer:in_buf      offset:0     atIndex:3];
    [enc setBuffer:out_buf     offset:0     atIndex:4];
    [enc setBytes:&out_dim     length:4     atIndex:5];
    [enc setBytes:&in_dim      length:4     atIndex:6];
    [enc setBytes:&group_size  length:4     atIndex:7];

    if (use_v3) {
        uint32_t num_tgs = (out_dim + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        [enc dispatchThreadgroups:MTLSizeMake(out_dim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
    [enc endEncoding];
}

// Encode one expert forward using multi-expert slot k.
// Expert data must already be in buf_multi_expert_data[k].
// Input must already be in buf_multi_expert_input.
static void gpu_encode_expert_forward_slot(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    int k  // slot index
) {
    NSUInteger gate_w_off, gate_s_off, gate_b_off;
    NSUInteger up_w_off, up_s_off, up_b_off;
    NSUInteger down_w_off, down_s_off, down_b_off;
    if (g_use_2bit) {
        gate_w_off = cfg.gate_w_off_2; gate_s_off = cfg.gate_s_off_2; gate_b_off = cfg.gate_b_off_2;
        up_w_off   = cfg.up_w_off_2;   up_s_off   = cfg.up_s_off_2;   up_b_off   = cfg.up_b_off_2;
        down_w_off = cfg.down_w_off_2; down_s_off = cfg.down_s_off_2; down_b_off = cfg.down_b_off_2;
    } else {
        gate_w_off = cfg.gate_w_off_4; gate_s_off = cfg.gate_s_off_4; gate_b_off = cfg.gate_b_off_4;
        up_w_off   = cfg.up_w_off_4;   up_s_off   = cfg.up_s_off_4;   up_b_off   = cfg.up_b_off_4;
        down_w_off = cfg.down_w_off_4;  down_s_off = cfg.down_s_off_4;  down_b_off = cfg.down_b_off_4;
    }
    id<MTLComputePipelineState> expert_pipe = g_use_2bit ? ctx->matvec_2bit : ctx->matvec_v3;

    uint32_t gate_up_out = cfg.moe_intermediate;
    uint32_t gate_up_in  = cfg.hidden_dim;
    uint32_t down_out    = cfg.hidden_dim;
    uint32_t down_in     = cfg.moe_intermediate;
    uint32_t gs          = cfg.group_size;

    // gate_proj: data[k] -> gate[k]
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:gate_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:gate_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:gate_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_gate[k]   offset:0           atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // up_proj: data[k] -> up[k]
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:up_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:up_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:up_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_input     offset:0          atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_up[k]     offset:0          atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // SwiGLU: gate[k], up[k] -> act[k]
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_up[k]   offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_act[k]  offset:0 atIndex:2];
        [enc setBytes:&gate_up_out length:4 atIndex:3];
        uint32_t swiglu_tgs = (gate_up_out + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // down_proj: act[k] -> out[k]
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:down_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:down_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:down_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_act[k]  offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_out[k]  offset:0           atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        uint32_t num_tgs = (down_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
}

// Encode one expert forward using explicit data buffer (for double buffering).
// Expert data must already be in data_buf.
// Input must already be in buf_multi_expert_input.
// Uses slot k's gate/up/act/out scratch buffers.
static void gpu_encode_expert_forward_slot_buf(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    int k,                  // slot index (for gate/up/act/out scratch)
    id<MTLBuffer> data_buf  // expert weight data buffer (from either set A or B)
) {
    NSUInteger gate_w_off, gate_s_off, gate_b_off;
    NSUInteger up_w_off, up_s_off, up_b_off;
    NSUInteger down_w_off, down_s_off, down_b_off;
    if (g_use_2bit) {
        gate_w_off = cfg.gate_w_off_2; gate_s_off = cfg.gate_s_off_2; gate_b_off = cfg.gate_b_off_2;
        up_w_off   = cfg.up_w_off_2;   up_s_off   = cfg.up_s_off_2;   up_b_off   = cfg.up_b_off_2;
        down_w_off = cfg.down_w_off_2; down_s_off = cfg.down_s_off_2; down_b_off = cfg.down_b_off_2;
    } else {
        gate_w_off = cfg.gate_w_off_4; gate_s_off = cfg.gate_s_off_4; gate_b_off = cfg.gate_b_off_4;
        up_w_off   = cfg.up_w_off_4;   up_s_off   = cfg.up_s_off_4;   up_b_off   = cfg.up_b_off_4;
        down_w_off = cfg.down_w_off_4;  down_s_off = cfg.down_s_off_4;  down_b_off = cfg.down_b_off_4;
    }
    id<MTLComputePipelineState> expert_pipe = g_use_2bit ? ctx->matvec_2bit : ctx->matvec_v3;

    uint32_t gate_up_out = cfg.moe_intermediate;
    uint32_t gate_up_in  = cfg.hidden_dim;
    uint32_t down_out    = cfg.hidden_dim;
    uint32_t down_in     = cfg.moe_intermediate;
    uint32_t gs          = cfg.group_size;

    // gate_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:data_buf                        offset:gate_w_off  atIndex:0];
        [enc setBuffer:data_buf                        offset:gate_s_off  atIndex:1];
        [enc setBuffer:data_buf                        offset:gate_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_gate[k]   offset:0           atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // up_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:data_buf                        offset:up_w_off  atIndex:0];
        [enc setBuffer:data_buf                        offset:up_s_off  atIndex:1];
        [enc setBuffer:data_buf                        offset:up_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_input     offset:0          atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_up[k]     offset:0          atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // SwiGLU
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_up[k]   offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_act[k]  offset:0 atIndex:2];
        [enc setBytes:&gate_up_out length:4 atIndex:3];
        uint32_t swiglu_tgs = (gate_up_out + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // down_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:data_buf                        offset:down_w_off  atIndex:0];
        [enc setBuffer:data_buf                        offset:down_s_off  atIndex:1];
        [enc setBuffer:data_buf                        offset:down_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_act[k]    offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_out[k]    offset:0           atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        uint32_t num_tgs = (down_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
}

// Batched expert encoding: encode K experts using 2 encoders per expert
// (gate+up fused, SwiGLU+down fused) + 2 for shared = K*2 + 2 encoders total.
// With K=4: 10 encoders (vs. old 4*K + 2 = 18 with per-operation encoding).
// Each expert gets its own encoder pair for GPU parallelism across experts.
// Within each encoder, gate+up (or SwiGLU+down) are serialized but share
// encoder creation overhead. Net win: fewer encoders, same parallelism.
static void gpu_encode_experts_batched(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    int K,                       // number of experts to encode
    const int *valid,            // which experts are valid [MAX_K]
    id<MTLBuffer> __strong *expert_bufs,   // per-expert weight data buffers [MAX_K]
    int layer_idx,               // layer index (for tiered manifest lookup)
    const int *expert_indices    // expert indices (for tiered per-expert quant)
) {
    uint32_t gate_up_out = cfg.moe_intermediate;
    uint32_t gate_up_in  = cfg.hidden_dim;
    uint32_t down_out    = cfg.hidden_dim;
    uint32_t down_in     = cfg.moe_intermediate;
    uint32_t gs          = cfg.group_size;
    // Threadgroup count is the same for 2-bit and 4-bit (based on out_dim).
    // The kernel handles packed_cols internally.
    uint32_t gate_up_tgs = (gate_up_out + 7) / 8;
    uint32_t down_tgs    = (down_out + 7) / 8;
    uint32_t swiglu_tgs  = (gate_up_out + 255) / 256;

    // Per-expert: Encoder A (gate+up), Encoder B (SwiGLU+down)
    // Separate encoders per expert enables GPU parallelism across experts.
    // Within each encoder, operations serialize (gate then up, SwiGLU then down).
    for (int k = 0; k < K; k++) {
        if (!valid[k]) continue;

        // Per-expert quantization selection (tiered: each expert may differ)
        int use_2bit_k;
        if (g_use_tiered && g_tiered_manifest) {
            use_2bit_k = (TIERED(layer_idx, expert_indices[k]).bits == 2);
        } else {
            use_2bit_k = g_use_2bit;
        }

        NSUInteger gate_w_off, gate_s_off, gate_b_off;
        NSUInteger up_w_off, up_s_off, up_b_off;
        NSUInteger down_w_off, down_s_off, down_b_off;
        id<MTLComputePipelineState> expert_pipe;

        if (use_2bit_k) {
            gate_w_off = cfg.gate_w_off_2; gate_s_off = cfg.gate_s_off_2; gate_b_off = cfg.gate_b_off_2;
            up_w_off   = cfg.up_w_off_2;   up_s_off   = cfg.up_s_off_2;   up_b_off   = cfg.up_b_off_2;
            down_w_off = cfg.down_w_off_2; down_s_off = cfg.down_s_off_2; down_b_off = cfg.down_b_off_2;
            expert_pipe = (g_use_fp16_accum && ctx->matvec_2bit_fp16) ? ctx->matvec_2bit_fp16 : ctx->matvec_2bit;
        } else {
            gate_w_off = cfg.gate_w_off_4; gate_s_off = cfg.gate_s_off_4; gate_b_off = cfg.gate_b_off_4;
            up_w_off   = cfg.up_w_off_4;   up_s_off   = cfg.up_s_off_4;   up_b_off   = cfg.up_b_off_4;
            down_w_off = cfg.down_w_off_4; down_s_off = cfg.down_s_off_4; down_b_off = cfg.down_b_off_4;
            expert_pipe = (g_use_fp16_accum && ctx->matvec_v3_fp16) ? ctx->matvec_v3_fp16 : ctx->matvec_v3;
        }

        // 4-bit path: fused gate+up+SwiGLU kernel (1 dispatch instead of 3)
        // 2-bit path: fallback to separate gate, up, SwiGLU dispatches
        if (!use_2bit_k && g_fused_expert_enabled && ctx->fused_gate_up) {
            // Encoder A: fused_gate_up_swiglu -> act[k] directly
            // NOTE: fused kernel uses 1 TG per output row (like matvec_fast),
            // NOT ROWS_PER_TG=8 rows per threadgroup (like matvec_v3).
            {
                id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
                id<MTLComputePipelineState> fused_pipe = (g_use_fp16_accum && ctx->fused_gate_up_fp16)
                    ? ctx->fused_gate_up_fp16 : ctx->fused_gate_up;
                [enc setComputePipelineState:fused_pipe];
                [enc setBuffer:expert_bufs[k]                  offset:gate_w_off  atIndex:0];
                [enc setBuffer:expert_bufs[k]                  offset:gate_s_off  atIndex:1];
                [enc setBuffer:expert_bufs[k]                  offset:gate_b_off  atIndex:2];
                [enc setBuffer:expert_bufs[k]                  offset:up_w_off    atIndex:3];
                [enc setBuffer:expert_bufs[k]                  offset:up_s_off    atIndex:4];
                [enc setBuffer:expert_bufs[k]                  offset:up_b_off    atIndex:5];
                [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:6];
                [enc setBuffer:ctx->buf_multi_expert_act[k]    offset:0           atIndex:7];
                [enc setBytes:&gate_up_out length:4 atIndex:8];
                [enc setBytes:&gate_up_in  length:4 atIndex:9];
                [enc setBytes:&gs          length:4 atIndex:10];
                [enc dispatchThreadgroups:MTLSizeMake(gate_up_out, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }

            // Encoder B: down_proj only (reads from act[k])
            {
                id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
                [enc setComputePipelineState:expert_pipe];
                [enc setBuffer:expert_bufs[k]                  offset:down_w_off  atIndex:0];
                [enc setBuffer:expert_bufs[k]                  offset:down_s_off  atIndex:1];
                [enc setBuffer:expert_bufs[k]                  offset:down_b_off  atIndex:2];
                [enc setBuffer:ctx->buf_multi_expert_act[k]    offset:0           atIndex:3];
                [enc setBuffer:ctx->buf_multi_expert_out[k]    offset:0           atIndex:4];
                [enc setBytes:&down_out length:4 atIndex:5];
                [enc setBytes:&down_in  length:4 atIndex:6];
                [enc setBytes:&gs       length:4 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(down_tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
        } else {
            // Fallback: separate gate + up + SwiGLU (for 2-bit or if fused pipeline unavailable)
            // Encoder A: gate_proj + up_proj (both read same input, write different outputs)
            {
                id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
                // gate_proj
                [enc setComputePipelineState:expert_pipe];
                [enc setBuffer:expert_bufs[k]                  offset:gate_w_off  atIndex:0];
                [enc setBuffer:expert_bufs[k]                  offset:gate_s_off  atIndex:1];
                [enc setBuffer:expert_bufs[k]                  offset:gate_b_off  atIndex:2];
                [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:3];
                [enc setBuffer:ctx->buf_multi_expert_gate[k]   offset:0           atIndex:4];
                [enc setBytes:&gate_up_out length:4 atIndex:5];
                [enc setBytes:&gate_up_in  length:4 atIndex:6];
                [enc setBytes:&gs          length:4 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(gate_up_tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                // up_proj (same encoder, serialized after gate — shares encoder overhead)
                [enc setBuffer:expert_bufs[k]                  offset:up_w_off  atIndex:0];
                [enc setBuffer:expert_bufs[k]                  offset:up_s_off  atIndex:1];
                [enc setBuffer:expert_bufs[k]                  offset:up_b_off  atIndex:2];
                [enc setBuffer:ctx->buf_multi_expert_up[k]     offset:0          atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(gate_up_tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }

            // Encoder B: SwiGLU + down_proj (SwiGLU depends on gate+up from Enc A)
            {
                id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
                // SwiGLU
                [enc setComputePipelineState:ctx->swiglu];
                [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_multi_expert_up[k]   offset:0 atIndex:1];
                [enc setBuffer:ctx->buf_multi_expert_act[k]  offset:0 atIndex:2];
                [enc setBytes:&gate_up_out length:4 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                // down_proj (same encoder, serialized after SwiGLU)
                [enc setComputePipelineState:expert_pipe];
            [enc setBuffer:expert_bufs[k]                  offset:down_w_off  atIndex:0];
            [enc setBuffer:expert_bufs[k]                  offset:down_s_off  atIndex:1];
            [enc setBuffer:expert_bufs[k]                  offset:down_b_off  atIndex:2];
            [enc setBuffer:ctx->buf_multi_expert_act[k]    offset:0           atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_out[k]    offset:0           atIndex:4];
            [enc setBytes:&down_out length:4 atIndex:5];
            [enc setBytes:&down_in  length:4 atIndex:6];
            [enc setBytes:&gs       length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(down_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            }
        }
    }
}

// Encode one expert forward (gate+up+swiglu+down) into cmdbuf.
// Expert data must already be in buf_expert_data.
// Input must already be in buf_expert_input.
__attribute__((unused))
static void gpu_encode_expert_forward(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf
) {
    NSUInteger gate_w_off = cfg.gate_w_off_4;
    NSUInteger gate_s_off = cfg.gate_s_off_4;
    NSUInteger gate_b_off = cfg.gate_b_off_4;
    NSUInteger up_w_off   = cfg.up_w_off_4;
    NSUInteger up_s_off   = cfg.up_s_off_4;
    NSUInteger up_b_off   = cfg.up_b_off_4;
    NSUInteger down_w_off = cfg.down_w_off_4;
    NSUInteger down_s_off = cfg.down_s_off_4;
    NSUInteger down_b_off = cfg.down_b_off_4;

    uint32_t gate_up_out = cfg.moe_intermediate;
    uint32_t gate_up_in  = cfg.hidden_dim;
    uint32_t down_out    = cfg.hidden_dim;
    uint32_t down_in     = cfg.moe_intermediate;
    uint32_t gs          = cfg.group_size;

    // gate_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->matvec_v3];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_input offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_expert_gate  offset:0           atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // up_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->matvec_v3];
        [enc setBuffer:ctx->buf_expert_data  offset:up_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data  offset:up_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data  offset:up_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_input offset:0          atIndex:3];
        [enc setBuffer:ctx->buf_expert_up    offset:0          atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // SwiGLU
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_expert_gate offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_expert_up   offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_expert_act  offset:0 atIndex:2];
        [enc setBytes:&gate_up_out length:4 atIndex:3];
        uint32_t swiglu_tgs = (gate_up_out + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // down_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->matvec_v3];
        [enc setBuffer:ctx->buf_expert_data offset:down_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data offset:down_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data offset:down_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_act  offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_expert_out  offset:0           atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        uint32_t num_tgs = (down_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
}

// Batched wrapper: takes N matmul specs sharing the same input, dispatches
// via GPU batch if available, otherwise falls back to CPU.
static void fast_batch_matvec(
    const float *x, uint32_t x_dim,
    BatchMatvecSpec *specs, int num_specs
) {
    if (g_metal && (g_metal->wf_num_chunks > 0 || g_metal->wf_staging)) {
        gpu_batch_matvec(g_metal, x, x_dim, specs, num_specs);
    } else {
        for (int i = 0; i < num_specs; i++) {
            BatchMatvecSpec *s = &specs[i];
            cpu_dequant_matvec(s->W, s->scales, s->biases, x, s->out_cpu,
                               s->out_dim, s->in_dim, s->group_size);
        }
    }
}

// ============================================================================
// GPU expert forward: gate+up matvec -> SwiGLU -> down matvec
// All 3 matmuls + activation in a single command buffer submission.
// Expert data is copied into a reusable Metal buffer.
// ============================================================================

// expert_data_already_in_buffer: if true, expert data is already in buf_expert_data
//   (pread'd directly into it), skip the copy.
__attribute__((unused))
static void gpu_expert_forward(
    MetalCtx *ctx,
    const void *expert_data,     // cfg.expert_size_4bit bytes (may be buf_expert_data contents)
    const float *h_post,         // [cfg.hidden_dim] input
    float *expert_out,           // [cfg.hidden_dim] output
    int expert_data_already_in_buffer
) {
    // Expert layout offsets — select based on quantization mode
    NSUInteger gate_w_off, gate_s_off, gate_b_off;
    NSUInteger up_w_off, up_s_off, up_b_off;
    NSUInteger down_w_off, down_s_off, down_b_off;
    if (g_use_2bit) {
        gate_w_off = cfg.gate_w_off_2; gate_s_off = cfg.gate_s_off_2; gate_b_off = cfg.gate_b_off_2;
        up_w_off   = cfg.up_w_off_2;   up_s_off   = cfg.up_s_off_2;   up_b_off   = cfg.up_b_off_2;
        down_w_off = cfg.down_w_off_2; down_s_off = cfg.down_s_off_2; down_b_off = cfg.down_b_off_2;
    } else {
        gate_w_off = cfg.gate_w_off_4; gate_s_off = cfg.gate_s_off_4; gate_b_off = cfg.gate_b_off_4;
        up_w_off   = cfg.up_w_off_4;   up_s_off   = cfg.up_s_off_4;   up_b_off   = cfg.up_b_off_4;
        down_w_off = cfg.down_w_off_4;  down_s_off = cfg.down_s_off_4;  down_b_off = cfg.down_b_off_4;
    }
    id<MTLComputePipelineState> expert_pipe = g_use_2bit ? ctx->matvec_2bit : ctx->matvec_v3;

    // Copy expert weights into Metal buffer only if not already there
    if (!expert_data_already_in_buffer) {
        memcpy([ctx->buf_expert_data contents], expert_data, active_expert_size());
    }
    memcpy([ctx->buf_expert_input contents], h_post, cfg.hidden_dim * sizeof(float));

    uint32_t gate_up_out = cfg.moe_intermediate;  // 1024
    uint32_t gate_up_in  = cfg.hidden_dim;        // 4096
    uint32_t down_out    = cfg.hidden_dim;        // 4096
    uint32_t down_in     = cfg.moe_intermediate;  // 1024
    uint32_t gs          = cfg.group_size;        // 64

    // Build one command buffer with all 4 dispatches:
    // 1. gate_proj matvec (h_post -> gate_out)
    // 2. up_proj matvec (h_post -> up_out)
    // 3. SwiGLU (gate_out, up_out -> act_out)
    // 4. down_proj matvec (act_out -> expert_out)

    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];

    // --- Dispatch 1: gate_proj [4096] -> [1024] ---
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_input offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_expert_gate  offset:0           atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    // --- Dispatch 2: up_proj [4096] -> [1024] ---
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_expert_data  offset:up_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data  offset:up_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data  offset:up_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_input offset:0          atIndex:3];
        [enc setBuffer:ctx->buf_expert_up    offset:0          atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    // --- Dispatch 3: SwiGLU(gate, up) -> act ---
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_expert_gate offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_expert_up   offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_expert_act  offset:0 atIndex:2];
        [enc setBytes:&gate_up_out length:4 atIndex:3];
        uint32_t swiglu_tgs = (gate_up_out + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    // --- Dispatch 4: down_proj [1024] -> [4096] ---
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_expert_data offset:down_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data offset:down_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data offset:down_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_act  offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_expert_out  offset:0           atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        uint32_t num_tgs = (down_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    // Copy result back to CPU
    memcpy(expert_out, [ctx->buf_expert_out contents], cfg.hidden_dim * sizeof(float));
}

// ============================================================================
// Rotary position embedding (for full attention layers)
// ============================================================================

static void apply_rotary_emb(float *q, float *k, int pos, int num_heads, int num_kv_heads,
                              int head_dim, int rotary_dim) {
    // Apply RoPE to the first rotary_dim dimensions of each head
    // NON-TRADITIONAL (MLX default): pairs are (x[i], x[i + half_dim])
    // where half_dim = rotary_dim / 2
    int half = rotary_dim / 2;
    for (int h = 0; h < num_heads; h++) {
        float *qh = q + h * head_dim;
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(cfg.rope_theta, (float)(2 * i) / rotary_dim);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            float q0 = qh[i];
            float q1 = qh[i + half];
            qh[i]        = q0 * cos_a - q1 * sin_a;
            qh[i + half]  = q0 * sin_a + q1 * cos_a;
        }
    }
    for (int h = 0; h < num_kv_heads; h++) {
        float *kh = k + h * head_dim;
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(cfg.rope_theta, (float)(2 * i) / rotary_dim);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            float k0 = kh[i];
            float k1 = kh[i + half];
            kh[i]        = k0 * cos_a - k1 * sin_a;
            kh[i + half]  = k0 * sin_a + k1 * cos_a;
        }
    }
}

// ============================================================================
// KV Cache for full attention layers
// ============================================================================

typedef struct KVCache_s {
    float *k_cache;      // [capacity, num_kv_heads * head_dim] (NULL when use_fp8=1)
    float *v_cache;      // [capacity, num_kv_heads * head_dim] (NULL when use_fp8=1)
    uint8_t *k_cache_fp8;  // [capacity, num_kv_heads * head_dim] FP8 E4M3 (NULL when use_fp8=0)
    uint8_t *v_cache_fp8;  // [capacity, num_kv_heads * head_dim] FP8 E4M3 (NULL when use_fp8=0)
    float *k_scales;     // [capacity] per-position K scale (FP8 only)
    float *v_scales;     // [capacity] per-position V scale (FP8 only)
    int len;             // total tokens written (monotonically increasing)
    int use_fp8;         // 1 = FP8 E4M3, 0 = float32
    int window_size;     // >0: sliding window (circular buffer), 0: unlimited
    int capacity;        // allocated size (= window_size if sliding, else max_seq)
    // H2O (Heavy Hitter Oracle) eviction state
    float *attn_scores_accum;  // [capacity] cumulative attention score per position
    int *token_positions;      // [capacity] original sequence position (for sink detection)
    int h2o_budget;           // total positions to keep (sinks + recent + heavy hitters)
    int h2o_num_sinks;        // number of sink tokens (first N, typically 4)
    int h2o_num_recent;       // number of recent tokens to always keep
    int h2o_active;           // 1 = H2O eviction enabled
    int h2o_num_valid;        // current number of valid positions in cache
    int *h2o_valid_indices;   // [capacity] indices of valid positions in contiguous order
} KVCache;

KVCache *kv_cache_new(void) {
    int max_seq = cfg.gpu_kv_seq > 0 ? cfg.gpu_kv_seq : cfg.max_seq_len;
    KVCache *c = calloc(1, sizeof(KVCache));
    c->len = 0;
    c->use_fp8 = g_use_fp8_kv;
    // H2O replaces sliding window when both are set (H2O is strictly better)
    if (g_h2o_budget > 0) {
        c->window_size = 0;  // disable sliding window — H2O handles eviction
    } else {
        c->window_size = g_sliding_window;  // 0 = unlimited, >0 = circular buffer
    }
    // Capacity: if sliding window, only allocate window_size positions
    // For H2O: allocate full budget (we compact in-place, never exceed budget+1)
    int seq;
    if (g_h2o_budget > 0) {
        seq = (g_h2o_budget + 1 < max_seq) ? g_h2o_budget + 1 : max_seq;
    } else if (c->window_size > 0 && c->window_size < max_seq) {
        seq = c->window_size;
    } else {
        seq = max_seq;
    }
    c->capacity = seq;
    size_t kv_dim = (size_t)cfg.num_kv_heads * cfg.head_dim;

    if (c->use_fp8) {
        c->k_cache_fp8 = calloc((size_t)seq * kv_dim, sizeof(uint8_t));
        c->v_cache_fp8 = calloc((size_t)seq * kv_dim, sizeof(uint8_t));
        c->k_scales = calloc(seq, sizeof(float));
        c->v_scales = calloc(seq, sizeof(float));
        c->k_cache = NULL;
        c->v_cache = NULL;
        if (!c->k_cache_fp8 || !c->v_cache_fp8 || !c->k_scales || !c->v_scales) {
            fprintf(stderr, "ERROR: FP8 KV cache alloc failed (seq=%d, %.1f MB each)\n",
                    seq, (double)seq * kv_dim * sizeof(uint8_t) / 1e6);
            free(c->k_cache_fp8); free(c->v_cache_fp8);
            free(c->k_scales); free(c->v_scales); free(c);
            return NULL;
        }
    } else {
        c->k_cache = calloc((size_t)seq * kv_dim, sizeof(float));
        c->v_cache = calloc((size_t)seq * kv_dim, sizeof(float));
        c->k_cache_fp8 = NULL; c->v_cache_fp8 = NULL;
        c->k_scales = NULL; c->v_scales = NULL;
        if (!c->k_cache || !c->v_cache) {
            fprintf(stderr, "ERROR: KV cache alloc failed (seq=%d, %.1f MB each)\n",
                    seq, (double)seq * kv_dim * sizeof(float) / 1e6);
            free(c->k_cache); free(c->v_cache); free(c);
            return NULL;
        }
    }
    // H2O initialization
    c->h2o_active = (g_h2o_budget > 0) ? 1 : 0;
    c->h2o_budget = g_h2o_budget;
    c->h2o_num_sinks = g_h2o_num_sinks;
    c->h2o_num_recent = c->h2o_active ? (c->h2o_budget - c->h2o_num_sinks) / 4 : 0;
    c->h2o_num_valid = 0;
    if (c->h2o_active) {
        c->attn_scores_accum = calloc(seq, sizeof(float));
        c->token_positions = calloc(seq, sizeof(int));
        c->h2o_valid_indices = calloc(seq, sizeof(int));
        if (!c->attn_scores_accum || !c->token_positions || !c->h2o_valid_indices) {
            fprintf(stderr, "ERROR: H2O alloc failed (seq=%d)\n", seq);
            free(c->attn_scores_accum); free(c->token_positions);
            free(c->h2o_valid_indices);
            c->h2o_active = 0;
        }
    } else {
        c->attn_scores_accum = NULL;
        c->token_positions = NULL;
        c->h2o_valid_indices = NULL;
    }
    return c;
}

static void kv_cache_free(KVCache *c) {
    if (c) {
        free(c->k_cache);
        free(c->v_cache);
        free(c->k_cache_fp8);
        free(c->v_cache_fp8);
        free(c->k_scales);
        free(c->v_scales);
        free(c->attn_scores_accum);
        free(c->token_positions);
        free(c->h2o_valid_indices);
        free(c);
    }
}

// ---- H2O (Heavy Hitter Oracle) KV cache eviction ----
// Evicts positions from the KV cache to keep at most h2o_budget entries.
// Protection categories: attention sinks (first N), recent tokens, heavy hitters.
// Called after each new KV write when H2O is active and num_valid exceeds budget.
static void kv_cache_evict_h2o(KVCache *kv) {
    if (!kv->h2o_active || kv->h2o_num_valid <= kv->h2o_budget) return;

    int budget = kv->h2o_budget;
    int num_sinks = kv->h2o_num_sinks;
    int num_recent = kv->h2o_num_recent;
    int num_valid = kv->h2o_num_valid;
    int kv_dim = cfg.num_kv_heads * cfg.head_dim;

    // Mark positions to keep: sinks + recent + heavy hitters
    int *keep = calloc(num_valid, sizeof(int));

    // 1. Always keep sink tokens (first num_sinks original positions)
    int kept = 0;
    for (int i = 0; i < num_valid && kept < num_sinks; i++) {
        if (kv->token_positions[i] < num_sinks) {
            keep[i] = 1;
            kept++;
        }
    }

    // 2. Always keep recent tokens (last num_recent written)
    for (int i = num_valid - num_recent; i < num_valid; i++) {
        if (i >= 0 && !keep[i]) {
            keep[i] = 1;
            kept++;
        }
    }

    // 3. Fill remaining budget with highest attention scores (heavy hitters)
    int hh_budget = budget - kept;
    if (hh_budget > 0) {
        // Find indices of non-kept positions sorted by score (simple selection)
        for (int h = 0; h < hh_budget; h++) {
            int best = -1;
            float best_score = -1e30f;
            for (int i = 0; i < num_valid; i++) {
                if (!keep[i] && kv->attn_scores_accum[i] > best_score) {
                    best_score = kv->attn_scores_accum[i];
                    best = i;
                }
            }
            if (best >= 0) {
                keep[best] = 1;
                kept++;
            }
        }
    }

    // Compact: move kept positions to front
    int write_pos = 0;
    for (int i = 0; i < num_valid; i++) {
        if (!keep[i]) continue;
        if (write_pos != i) {
            if (kv->use_fp8) {
                memcpy(kv->k_cache_fp8 + write_pos * kv_dim, kv->k_cache_fp8 + i * kv_dim, kv_dim);
                memcpy(kv->v_cache_fp8 + write_pos * kv_dim, kv->v_cache_fp8 + i * kv_dim, kv_dim);
                kv->k_scales[write_pos] = kv->k_scales[i];
                kv->v_scales[write_pos] = kv->v_scales[i];
            } else {
                memcpy(kv->k_cache + write_pos * kv_dim, kv->k_cache + i * kv_dim, kv_dim * sizeof(float));
                memcpy(kv->v_cache + write_pos * kv_dim, kv->v_cache + i * kv_dim, kv_dim * sizeof(float));
            }
            kv->attn_scores_accum[write_pos] = kv->attn_scores_accum[i];
            kv->token_positions[write_pos] = kv->token_positions[i];
        }
        write_pos++;
    }
    kv->h2o_num_valid = write_pos;
    free(keep);
}

// Sync GPU KV buffers after H2O compaction (re-upload compacted data)
static void kv_cache_h2o_sync_gpu(KVCache *kv, int fa_idx) {
    if (!kv->h2o_active || !g_metal || fa_idx < 0) return;
    int kv_dim = cfg.num_kv_heads * cfg.head_dim;
    int num_valid = kv->h2o_num_valid;

    if (g_use_fp8_kv && g_metal->buf_kv_k_scales && g_metal->buf_kv_k_scales[fa_idx]) {
        memcpy([g_metal->buf_kv_k[fa_idx] contents], kv->k_cache_fp8, (size_t)num_valid * kv_dim);
        memcpy([g_metal->buf_kv_v[fa_idx] contents], kv->v_cache_fp8, (size_t)num_valid * kv_dim);
        memcpy([g_metal->buf_kv_k_scales[fa_idx] contents], kv->k_scales, num_valid * sizeof(float));
        memcpy([g_metal->buf_kv_v_scales[fa_idx] contents], kv->v_scales, num_valid * sizeof(float));
    } else if (g_metal->buf_kv_k && g_metal->buf_kv_k[fa_idx]) {
        memcpy([g_metal->buf_kv_k[fa_idx] contents], kv->k_cache, (size_t)num_valid * kv_dim * sizeof(float));
        memcpy([g_metal->buf_kv_v[fa_idx] contents], kv->v_cache, (size_t)num_valid * kv_dim * sizeof(float));
    }
}

// ============================================================================
// Linear attention state (GatedDeltaNet recurrent state)
// ============================================================================

typedef struct LinearAttnState_s {
    float *conv_state;  // [(kernel_size-1) * conv_dim] for conv1d
    float *ssm_state;   // [num_v_heads, head_v_dim, head_k_dim] recurrent state
} LinearAttnState;

LinearAttnState *linear_attn_state_new(void) {
    LinearAttnState *s = calloc(1, sizeof(LinearAttnState));
    s->conv_state = calloc((cfg.conv_kernel_size - 1) * cfg.linear_conv_dim, sizeof(float));
    s->ssm_state = calloc(cfg.linear_num_v_heads * cfg.linear_value_dim * cfg.linear_key_dim, sizeof(float));
    return s;
}

static void linear_attn_state_free(LinearAttnState *s) {
    if (s) {
        free(s->conv_state);
        free(s->ssm_state);
        free(s);
    }
}

// ============================================================================
// Full attention layer forward (single token, incremental)
// ============================================================================

static int fa_debug_count = 0;

static float vec_rms(const float *v, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += v[i] * v[i];
    return sqrtf(sum / n);
}

__attribute__((unused))
static void full_attention_forward(
    WeightFile *wf,
    int layer_idx,
    float *hidden,       // [cfg.hidden_dim] in/out
    KVCache *kv,
    int pos              // position in sequence
) {
    fa_debug_count++;
    int do_debug = 0;  // set to (fa_debug_count <= N) to enable debug

    char name[256];
    float *normed = malloc(cfg.hidden_dim * sizeof(float));
    float *residual = malloc(cfg.hidden_dim * sizeof(float));
    cpu_vec_copy(residual, hidden, cfg.hidden_dim);

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] layer=%d pos=%d hidden_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, pos, vec_rms(hidden, cfg.hidden_dim),
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }

    // ---- Input LayerNorm ----
    snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer_idx);
    uint16_t *norm_w = get_tensor_ptr(wf, name);
    cpu_rms_norm(hidden, norm_w, normed, cfg.hidden_dim, cfg.rms_norm_eps);

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] normed_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(normed, cfg.hidden_dim), normed[0], normed[1], normed[2], normed[3], normed[4]);
    }

    // ---- QKV Projection ----
    // Qwen: Q projection outputs num_heads * head_dim * 2 (queries + sigmoid gate)
    // MiniMax: Q projection outputs num_heads * head_dim (queries only, no gate)
    int q_dim = cfg.num_attn_heads * cfg.head_dim;
    int q_proj_dim = cfg.has_attn_gate ? q_dim * 2 : q_dim;
    int kv_dim = cfg.num_kv_heads * cfg.head_dim;

    float *q_proj_out = calloc(q_proj_dim, sizeof(float));
    float *k = calloc(kv_dim, sizeof(float));
    float *v = calloc(kv_dim, sizeof(float));

    // Batch Q/K/V projections into a single GPU command buffer
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", layer_idx);
    uint32_t *qw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.scales", layer_idx);
    uint16_t *qs = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.biases", layer_idx);
    uint16_t *qb = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", layer_idx);
    uint32_t *kw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.scales", layer_idx);
    uint16_t *ks = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.biases", layer_idx);
    uint16_t *kb = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", layer_idx);
    uint32_t *vw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.scales", layer_idx);
    uint16_t *vs = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.biases", layer_idx);
    uint16_t *vb = get_tensor_ptr(wf, name);

    // Batch Q/K/V into one command buffer (3 dispatches, 1 commit)
    if (qw && qs && qb && kw && ks && kb && vw && vs && vb) {
        BatchMatvecSpec qkv_specs[3] = {
            { qw, qs, qb, q_proj_out, (uint32_t)q_proj_dim, cfg.hidden_dim, cfg.group_size, 0 },
            { kw, ks, kb, k,          (uint32_t)kv_dim,     cfg.hidden_dim, cfg.group_size, 1 },
            { vw, vs, vb, v,          (uint32_t)kv_dim,     cfg.hidden_dim, cfg.group_size, 2 },
        };
        fast_batch_matvec(normed, cfg.hidden_dim, qkv_specs, 3);
    }

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] q_proj first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                q_proj_out[0], q_proj_out[1], q_proj_out[2], q_proj_out[3], q_proj_out[4]);
    }

    // Split q_proj_out into queries and gate (Qwen only)
    float *q = calloc(q_dim, sizeof(float));
    float *q_gate = NULL;
    if (cfg.has_attn_gate) {
        q_gate = calloc(q_dim, sizeof(float));
        for (int h = 0; h < cfg.num_attn_heads; h++) {
            float *src = q_proj_out + h * (2 * cfg.head_dim);
            memcpy(q + h * cfg.head_dim, src, cfg.head_dim * sizeof(float));
            memcpy(q_gate + h * cfg.head_dim, src + cfg.head_dim, cfg.head_dim * sizeof(float));
        }
    } else {
        memcpy(q, q_proj_out, q_dim * sizeof(float));
    }
    free(q_proj_out);

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] v_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(v, kv_dim), v[0], v[1], v[2], v[3], v[4]);
        if (q_gate) {
            fprintf(stderr, "[FA-DBG] q_gate_rms=%.6f gate_first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                    vec_rms(q_gate, q_dim), q_gate[0], q_gate[1], q_gate[2], q_gate[3], q_gate[4]);
            float gate_sigmoid_sum = 0.0f;
            for (int i = 0; i < q_dim; i++) {
                gate_sigmoid_sum += 1.0f / (1.0f + expf(-q_gate[i]));
            }
            fprintf(stderr, "[FA-DBG] gate_sigmoid_mean=%.6f\n", gate_sigmoid_sum / q_dim);
        }
    }

    // ---- Q/K RMSNorm ----
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", layer_idx);
    uint16_t *qnorm_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", layer_idx);
    uint16_t *knorm_w = get_tensor_ptr(wf, name);

    // Apply Q norm
    if (qnorm_w) {
        if (cfg.qk_norm_per_layer) {
            // MiniMax: RMSNorm over entire flat q vector [num_heads * head_dim]
            float sum_sq = 0.0f;
            for (int i = 0; i < q_dim; i++) sum_sq += q[i] * q[i];
            float inv_rms = 1.0f / sqrtf(sum_sq / q_dim + cfg.rms_norm_eps);
            for (int i = 0; i < q_dim; i++) q[i] = q[i] * inv_rms * bf16_to_f32(qnorm_w[i]);
        } else {
            // Qwen: RMSNorm per head, weight is [head_dim] shared across heads
            for (int h = 0; h < cfg.num_attn_heads; h++) {
                float *qh = q + h * cfg.head_dim;
                float sum_sq = 0.0f;
                for (int i = 0; i < cfg.head_dim; i++) sum_sq += qh[i] * qh[i];
                float inv_rms = 1.0f / sqrtf(sum_sq / cfg.head_dim + cfg.rms_norm_eps);
                for (int i = 0; i < cfg.head_dim; i++) qh[i] = qh[i] * inv_rms * bf16_to_f32(qnorm_w[i]);
            }
        }
    }
    // Apply K norm
    if (knorm_w) {
        if (cfg.qk_norm_per_layer) {
            // MiniMax: RMSNorm over entire flat k vector
            int kv_dim_local = cfg.num_kv_heads * cfg.head_dim;
            float sum_sq = 0.0f;
            for (int i = 0; i < kv_dim_local; i++) sum_sq += k[i] * k[i];
            float inv_rms = 1.0f / sqrtf(sum_sq / kv_dim_local + cfg.rms_norm_eps);
            for (int i = 0; i < kv_dim_local; i++) k[i] = k[i] * inv_rms * bf16_to_f32(knorm_w[i]);
        } else {
            // Qwen: RMSNorm per head
            for (int h = 0; h < cfg.num_kv_heads; h++) {
                float *kh = k + h * cfg.head_dim;
                float sum_sq = 0.0f;
                for (int i = 0; i < cfg.head_dim; i++) sum_sq += kh[i] * kh[i];
                float inv_rms = 1.0f / sqrtf(sum_sq / cfg.head_dim + cfg.rms_norm_eps);
                for (int i = 0; i < cfg.head_dim; i++) kh[i] = kh[i] * inv_rms * bf16_to_f32(knorm_w[i]);
            }
        }
    }


    // ---- RoPE ----
    apply_rotary_emb(q, k, pos, cfg.num_attn_heads, cfg.num_kv_heads, cfg.head_dim, cfg.rotary_dim);

    // ---- Update KV cache (circular buffer for sliding window, or H2O) ----
    int cache_pos;
    if (kv->h2o_active) {
        cache_pos = kv->h2o_num_valid;
        if (cache_pos >= kv->capacity) {
            fprintf(stderr, "ERROR: H2O KV cache overflow (pos=%d >= cap=%d)\n", cache_pos, kv->capacity);
            free(normed); free(residual); free(q); free(q_gate); free(k); free(v);
            return;
        }
    } else if (kv->window_size > 0) {
        cache_pos = kv->len % kv->capacity;  // circular write
    } else {
        cache_pos = kv->len;
        if (cache_pos >= kv->capacity) {
            fprintf(stderr, "ERROR: KV cache overflow (pos=%d >= cap=%d)\n", cache_pos, kv->capacity);
            free(normed); free(residual); free(q); free(q_gate); free(k); free(v);
            return;
        }
    }
    if (kv->use_fp8) {
        if (kv->k_cache_fp8 && kv->v_cache_fp8) {
            kv->k_scales[cache_pos] = fp8_encode_vec(k, kv->k_cache_fp8 + cache_pos * kv_dim, kv_dim);
            kv->v_scales[cache_pos] = fp8_encode_vec(v, kv->v_cache_fp8 + cache_pos * kv_dim, kv_dim);
        }
    } else {
        memcpy(kv->k_cache + cache_pos * kv_dim, k, kv_dim * sizeof(float));
        memcpy(kv->v_cache + cache_pos * kv_dim, v, kv_dim * sizeof(float));
    }
    if (kv->h2o_active) {
        kv->token_positions[cache_pos] = kv->len;
        kv->attn_scores_accum[cache_pos] = 0.0f;
        kv->h2o_num_valid++;
    }
    kv->len++;

    // ---- Scaled dot-product attention ----
    // GQA: cfg.num_attn_heads=32 heads, cfg.num_kv_heads=2 kv heads
    // Each group of 16 query heads shares 1 kv head
    int heads_per_kv = cfg.num_attn_heads / cfg.num_kv_heads;
    float scale = 1.0f / sqrtf((float)cfg.head_dim);

    float *attn_out = calloc(q_dim, sizeof(float));

    // Temp buffer for dequantized K/V when using FP8
    float *k_dequant = kv->use_fp8 ? malloc(kv_dim * sizeof(float)) : NULL;
    float *v_dequant = kv->use_fp8 ? malloc(kv_dim * sizeof(float)) : NULL;

    // Determine attention range
    int attn_len;
    if (kv->h2o_active) {
        attn_len = kv->h2o_num_valid;
    } else if (kv->window_size > 0 && kv->len > kv->window_size) {
        attn_len = kv->window_size;
    } else {
        attn_len = kv->len;
    }

    // H2O score accumulator for this step
    float *h2o_step_scores = kv->h2o_active ? calloc(attn_len, sizeof(float)) : NULL;

    for (int h = 0; h < cfg.num_attn_heads; h++) {
        int kv_h = h / heads_per_kv;
        float *qh = q + h * cfg.head_dim;

        float *scores = malloc(attn_len * sizeof(float));
        if (!scores) {
            fprintf(stderr, "ERROR: attention scores alloc failed (len=%d)\n", attn_len);
            continue;
        }
        for (int i = 0; i < attn_len; i++) {
            int p;
            if (!kv->h2o_active && kv->window_size > 0 && kv->len > kv->window_size) {
                p = (kv->len - kv->window_size + i) % kv->capacity;
            } else {
                p = i;
            }
            float dot = 0.0f;
            if (kv->use_fp8) {
                fp8_decode_vec(kv->k_cache_fp8 + p * kv_dim, k_dequant, kv_dim, kv->k_scales[p]);
                float *kp = k_dequant + kv_h * cfg.head_dim;
                for (int d = 0; d < cfg.head_dim; d++) dot += qh[d] * kp[d];
            } else {
                float *kp = kv->k_cache + p * kv_dim + kv_h * cfg.head_dim;
                for (int d = 0; d < cfg.head_dim; d++) dot += qh[d] * kp[d];
            }
            scores[i] = dot * scale;
        }

        cpu_softmax(scores, attn_len);

        // Accumulate softmax scores for H2O
        if (h2o_step_scores) {
            for (int i = 0; i < attn_len; i++) h2o_step_scores[i] += scores[i];
        }

        float *oh = attn_out + h * cfg.head_dim;
        for (int i = 0; i < attn_len; i++) {
            int p;
            if (!kv->h2o_active && kv->window_size > 0 && kv->len > kv->window_size) {
                p = (kv->len - kv->window_size + i) % kv->capacity;
            } else {
                p = i;
            }
            if (kv->use_fp8) {
                fp8_decode_vec(kv->v_cache_fp8 + p * kv_dim, v_dequant, kv_dim, kv->v_scales[p]);
                float *vp = v_dequant + kv_h * cfg.head_dim;
                for (int d = 0; d < cfg.head_dim; d++) oh[d] += scores[i] * vp[d];
            } else {
                float *vp = kv->v_cache + p * kv_dim + kv_h * cfg.head_dim;
                for (int d = 0; d < cfg.head_dim; d++) oh[d] += scores[i] * vp[d];
            }
        }
        free(scores);
    }
    // Update H2O cumulative scores and run eviction
    if (h2o_step_scores) {
        for (int i = 0; i < attn_len; i++) kv->attn_scores_accum[i] += h2o_step_scores[i];
        free(h2o_step_scores);
        kv_cache_evict_h2o(kv);
    }
    free(k_dequant);
    free(v_dequant);


    // ---- Apply sigmoid gate to attention output (Qwen only) ----
    // MLX: return self.o_proj(output * mx.sigmoid(gate))
    if (q_gate) {
        for (int i = 0; i < q_dim; i++) {
            float g = 1.0f / (1.0f + expf(-q_gate[i]));
            attn_out[i] *= g;
        }
    }

    // ---- Output projection ----
    float *attn_projected = calloc(cfg.hidden_dim, sizeof(float));
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", layer_idx);
    uint32_t *ow = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.scales", layer_idx);
    uint16_t *os_ptr = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.biases", layer_idx);
    uint16_t *ob = get_tensor_ptr(wf, name);
    if (ow && os_ptr && ob) fast_dequant_matvec(ow, os_ptr, ob, attn_out, attn_projected, cfg.hidden_dim, q_dim, cfg.group_size);

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] attn_out_rms=%.6f o_proj first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(attn_out, q_dim),
                attn_projected[0], attn_projected[1], attn_projected[2], attn_projected[3], attn_projected[4]);
    }

    // ---- Residual connection ----
    for (int i = 0; i < cfg.hidden_dim; i++) {
        hidden[i] = residual[i] + attn_projected[i];
    }

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] AFTER layer=%d hidden_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, vec_rms(hidden, cfg.hidden_dim),
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }

    free(normed);
    free(residual);
    free(q);
    free(q_gate);
    free(k);
    free(v);
    free(attn_out);
    free(attn_projected);
}

// ============================================================================
// Linear attention layer forward (GatedDeltaNet, single token, incremental)
// ============================================================================

// RMS norm without weights (just normalize)
static void cpu_rms_norm_bare(const float *x, float *out, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / dim + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * inv_rms;
}

// RMSNormGated: out = rms_norm(x) * silu(z)
static void cpu_rms_norm_gated(const float *x, const float *z, const uint16_t *w_bf16,
                                float *out, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / dim + eps);
    for (int i = 0; i < dim; i++) {
        float w = bf16_to_f32(w_bf16[i]);
        float silu_z = z[i] / (1.0f + expf(-z[i]));
        out[i] = x[i] * inv_rms * w * silu_z;
    }
}

static int linear_attn_bypass = 0;  // set to 1 to skip linear attention (identity)
static int gpu_linear_attn_enabled = 1;  // fused GPU delta-net path (can disable via --cpu-linear)

__attribute__((unused))
static void linear_attention_forward(
    WeightFile *wf,
    int layer_idx,
    float *hidden,           // [cfg.hidden_dim] in/out
    LinearAttnState *state
) {
    // If bypass is enabled, just pass through (identity)
    if (linear_attn_bypass) {
        (void)wf; (void)layer_idx; (void)state;
        return;
    }

    static int la_debug_count = 0;
    la_debug_count++;
    int la_debug = 0;  // set to (la_debug_count <= N) to enable debug

    if (la_debug) {
        fprintf(stderr, "[LA-DBG] layer=%d hidden_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, vec_rms(hidden, cfg.hidden_dim),
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }

    char name[256];
    float *normed = malloc(cfg.hidden_dim * sizeof(float));
    float *residual = malloc(cfg.hidden_dim * sizeof(float));
    cpu_vec_copy(residual, hidden, cfg.hidden_dim);

    // ---- Input LayerNorm ----
    snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer_idx);
    uint16_t *norm_w = get_tensor_ptr(wf, name);
    cpu_rms_norm(hidden, norm_w, normed, cfg.hidden_dim, cfg.rms_norm_eps);

    // ---- Batch QKV + Z + B + A projections (4 matmuls, 1 command buffer) ----
    int qkv_dim = cfg.linear_conv_dim;  // 12288
    float *qkv = calloc(qkv_dim, sizeof(float));
    int z_dim = cfg.linear_total_value;  // 8192
    float *z = calloc(z_dim, sizeof(float));
    float *beta = calloc(cfg.linear_num_v_heads, sizeof(float));
    float *alpha = calloc(cfg.linear_num_v_heads, sizeof(float));

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.weight", layer_idx);
    uint32_t *qkv_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.scales", layer_idx);
    uint16_t *qkv_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.biases", layer_idx);
    uint16_t *qkv_b = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.weight", layer_idx);
    uint32_t *z_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.scales", layer_idx);
    uint16_t *z_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.biases", layer_idx);
    uint16_t *z_b = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.weight", layer_idx);
    uint32_t *b_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.scales", layer_idx);
    uint16_t *b_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.biases", layer_idx);
    uint16_t *b_b = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.weight", layer_idx);
    uint32_t *a_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.scales", layer_idx);
    uint16_t *a_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.biases", layer_idx);
    uint16_t *a_b = get_tensor_ptr(wf, name);

    if (qkv_w && qkv_s && qkv_b && z_w && z_s && z_b &&
        b_w && b_s && b_b && a_w && a_s && a_b) {
        BatchMatvecSpec la_specs[4] = {
            { qkv_w, qkv_s, qkv_b, qkv,   (uint32_t)qkv_dim,         cfg.hidden_dim, cfg.group_size, 0 },
            { z_w,   z_s,   z_b,   z,      (uint32_t)z_dim,           cfg.hidden_dim, cfg.group_size, 1 },
            { b_w,   b_s,   b_b,   beta,   (uint32_t)cfg.linear_num_v_heads, cfg.hidden_dim, cfg.group_size, 2 },
            { a_w,   a_s,   a_b,   alpha,  (uint32_t)cfg.linear_num_v_heads, cfg.hidden_dim, cfg.group_size, 3 },
        };
        fast_batch_matvec(normed, cfg.hidden_dim, la_specs, 4);
    }

    // ---- Conv1d step ----
    // conv_state holds last (kernel_size-1) inputs for each of the conv_dim channels
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.conv1d.weight", layer_idx);
    uint16_t *conv_w = get_tensor_ptr(wf, name);

    float *conv_out = calloc(qkv_dim, sizeof(float));
    if (conv_w) {
        cpu_conv1d_step(state->conv_state, qkv, conv_w, conv_out,
                        qkv_dim, cfg.conv_kernel_size);
    }

    // Update conv state: shift left, append new input
    memmove(state->conv_state, state->conv_state + qkv_dim,
            (cfg.conv_kernel_size - 2) * qkv_dim * sizeof(float));
    memcpy(state->conv_state + (cfg.conv_kernel_size - 2) * qkv_dim, qkv,
           qkv_dim * sizeof(float));

    // ---- Split conv_out into q, k, v ----
    // q: [num_k_heads * head_k_dim] = [2048]
    // k: [num_k_heads * head_k_dim] = [2048]
    // v: [num_v_heads * head_v_dim] = [8192]
    float *lin_q = conv_out;  // first cfg.linear_total_key elements
    float *lin_k = conv_out + cfg.linear_total_key;  // next cfg.linear_total_key
    float *lin_v = conv_out + 2 * cfg.linear_total_key;  // rest = cfg.linear_total_value

    // ---- RMS normalize q and k (bare, no weights) ----
    // q: scale = key_dim^(-0.5), normalize per head then scale by key_dim^(-1.0)
    // Actually from the code:
    //   inv_scale = k.shape[-1] ** -0.5 = head_k_dim^(-0.5) = 128^(-0.5)
    //   q = (inv_scale**2) * rms_norm(q) = (1/128) * rms_norm(q)
    //   k = inv_scale * rms_norm(k) = (1/sqrt(128)) * rms_norm(k)
    float inv_scale = 1.0f / sqrtf((float)cfg.linear_key_dim);

    for (int h = 0; h < cfg.linear_num_k_heads; h++) {
        float *qh = lin_q + h * cfg.linear_key_dim;
        cpu_rms_norm_bare(qh, qh, cfg.linear_key_dim, 1e-6f);
        float q_scale = inv_scale * inv_scale;  // inv_scale^2 = 1/head_k_dim
        for (int d = 0; d < cfg.linear_key_dim; d++) qh[d] *= q_scale;
    }
    for (int h = 0; h < cfg.linear_num_k_heads; h++) {
        float *kh = lin_k + h * cfg.linear_key_dim;
        cpu_rms_norm_bare(kh, kh, cfg.linear_key_dim, 1e-6f);
        for (int d = 0; d < cfg.linear_key_dim; d++) kh[d] *= inv_scale;
    }

    // ---- Gated delta net recurrence ----
    // From gated_delta.py:
    //   g = exp(-exp(A_log) * softplus(a + dt_bias))   -- per-head decay
    //   beta_gate = sigmoid(b)                          -- per-head beta (NO dt_bias)
    //   For each v_head:
    //     state = state * g                             -- decay
    //     kv_mem = sum(state * k, axis=key_dim)         -- predict v from state
    //     delta = (v - kv_mem) * beta_gate              -- error signal
    //     state = state + outer(delta, k)               -- update state
    //     output = sum(state * q, axis=key_dim)         -- read from state

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.A_log", layer_idx);
    float *A_log = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.dt_bias", layer_idx);
    uint16_t *dt_bias_bf16 = get_tensor_ptr(wf, name);

    float *out_values = calloc(cfg.linear_total_value, sizeof(float));  // [num_v_heads * head_v_dim]

    int k_heads_per_v = cfg.linear_num_v_heads / cfg.linear_num_k_heads;  // 64/16 = 4

    // Precompute per-head decay (g) and beta
    float g_decay[cfg.linear_num_v_heads];
    float beta_gate[cfg.linear_num_v_heads];
    for (int vh = 0; vh < cfg.linear_num_v_heads; vh++) {
        // g = exp(-exp(A_log) * softplus(a + dt_bias))
        float a_val = alpha[vh];
        float dt_b = dt_bias_bf16 ? bf16_to_f32(dt_bias_bf16[vh]) : 0.0f;
        float A_val = A_log ? expf(A_log[vh]) : 1.0f;
        float softplus_val = logf(1.0f + expf(a_val + dt_b));  // softplus(a + dt_bias)
        g_decay[vh] = expf(-A_val * softplus_val);

        // beta = sigmoid(b)  (just b, NO dt_bias)
        beta_gate[vh] = cpu_sigmoid(beta[vh]);
    }

    for (int vh = 0; vh < cfg.linear_num_v_heads; vh++) {
        int kh = vh / k_heads_per_v;  // which k head this v head maps to

        float g = g_decay[vh];
        float b_gate = beta_gate[vh];

        // state is [head_v_dim, head_k_dim]
        float *S = state->ssm_state + vh * cfg.linear_value_dim * cfg.linear_key_dim;
        float *v_h = lin_v + vh * cfg.linear_value_dim;
        float *k_h = lin_k + kh * cfg.linear_key_dim;

        // Step 1: Decay state
        for (int vi = 0; vi < cfg.linear_value_dim; vi++) {
            for (int ki = 0; ki < cfg.linear_key_dim; ki++) {
                S[vi * cfg.linear_key_dim + ki] *= g;
            }
        }

        // Step 2: Compute kv_mem[vi] = sum_ki(S[vi,ki] * k[ki])
        // Then delta[vi] = (v[vi] - kv_mem[vi]) * beta
        // Then state[vi,ki] += k[ki] * delta[vi]
        for (int vi = 0; vi < cfg.linear_value_dim; vi++) {
            float kv_mem = 0.0f;
            for (int ki = 0; ki < cfg.linear_key_dim; ki++) {
                kv_mem += S[vi * cfg.linear_key_dim + ki] * k_h[ki];
            }
            float delta = (v_h[vi] - kv_mem) * b_gate;
            for (int ki = 0; ki < cfg.linear_key_dim; ki++) {
                S[vi * cfg.linear_key_dim + ki] += k_h[ki] * delta;
            }
        }

        // Step 3: Output: y[vi] = sum_ki(S[vi,ki] * q[ki])
        float *q_h = lin_q + kh * cfg.linear_key_dim;
        float *o_h = out_values + vh * cfg.linear_value_dim;
        for (int vi = 0; vi < cfg.linear_value_dim; vi++) {
            float sum = 0.0f;
            for (int ki = 0; ki < cfg.linear_key_dim; ki++) {
                sum += S[vi * cfg.linear_key_dim + ki] * q_h[ki];
            }
            o_h[vi] = sum;
        }
    }

    // ---- RMSNormGated: out = rms_norm(out_values_per_head) * silu(z_per_head) * weight ----
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.norm.weight", layer_idx);
    uint16_t *gated_norm_w = get_tensor_ptr(wf, name);

    float *gated_out = calloc(cfg.linear_total_value, sizeof(float));
    for (int vh = 0; vh < cfg.linear_num_v_heads; vh++) {
        float *oh = out_values + vh * cfg.linear_value_dim;
        float *zh = z + vh * cfg.linear_value_dim;
        float *gh = gated_out + vh * cfg.linear_value_dim;
        if (gated_norm_w) {
            cpu_rms_norm_gated(oh, zh, gated_norm_w, gh, cfg.linear_value_dim, cfg.rms_norm_eps);
        } else {
            memcpy(gh, oh, cfg.linear_value_dim * sizeof(float));
        }
    }

    // ---- Output projection: [value_dim=8192] -> [hidden_dim=4096] ----
    float *attn_out = calloc(cfg.hidden_dim, sizeof(float));
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.weight", layer_idx);
    uint32_t *out_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.scales", layer_idx);
    uint16_t *out_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.biases", layer_idx);
    uint16_t *out_b = get_tensor_ptr(wf, name);
    if (out_w && out_s && out_b) {
        fast_dequant_matvec(out_w, out_s, out_b, gated_out, attn_out, cfg.hidden_dim,
                            cfg.linear_total_value, cfg.group_size);
    }

    // ---- Residual ----
    for (int i = 0; i < cfg.hidden_dim; i++) {
        hidden[i] = residual[i] + attn_out[i];
    }

    if (la_debug) {
        fprintf(stderr, "[LA-DBG] AFTER layer=%d out_proj_rms=%.6f gated_rms=%.6f hidden_rms=%.6f\n",
                layer_idx, vec_rms(attn_out, cfg.hidden_dim),
                vec_rms(gated_out, cfg.linear_total_value),
                vec_rms(hidden, cfg.hidden_dim));
    }

    free(normed);
    free(residual);
    free(qkv);
    free(z);
    free(beta);
    free(alpha);
    free(conv_out);
    free(out_values);
    free(gated_out);
    free(attn_out);
}

// ============================================================================
// MoE forward (routing + expert computation + shared expert)
// ============================================================================

static int moe_debug_count = 0;

__attribute__((unused))
static void moe_forward(
    WeightFile *wf,
    int layer_idx,
    float *hidden,         // [cfg.hidden_dim] in/out
    const char *model_path __attribute__((unused)),
    int K,                 // number of active experts (e.g. 4)
    int packed_fd          // fd for this layer's packed expert file (-1 if not available)
) {
    moe_debug_count++;
    int moe_debug = 0;  // set to (moe_debug_count <= N) to enable debug
    int moe_dump = 0;

    char name[256];
    float *h_post = malloc(cfg.hidden_dim * sizeof(float));
    float *h_mid = malloc(cfg.hidden_dim * sizeof(float));
    cpu_vec_copy(h_mid, hidden, cfg.hidden_dim);

    // ---- Post-attention LayerNorm ----
    snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", layer_idx);
    uint16_t *norm_w = get_tensor_ptr(wf, name);
    cpu_rms_norm(hidden, norm_w, h_post, cfg.hidden_dim, cfg.rms_norm_eps);

    // ---- Routing gate + (optionally) shared expert projections ----
    float *gate_scores = calloc(cfg.num_experts, sizeof(float));
    float *shared_gate = cfg.shared_intermediate > 0 ? calloc(cfg.shared_intermediate, sizeof(float)) : NULL;
    float *shared_up   = cfg.shared_intermediate > 0 ? calloc(cfg.shared_intermediate, sizeof(float)) : NULL;
    float shared_gate_score = 0.0f;

    snprintf(name, sizeof(name), "model.layers.%d.%s.gate.weight", layer_idx, cfg.moe_prefix);
    uint32_t *gate_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.%s.gate.scales", layer_idx, cfg.moe_prefix);
    uint16_t *gate_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.%s.gate.biases", layer_idx, cfg.moe_prefix);
    uint16_t *gate_b = get_tensor_ptr(wf, name);

    // Load routing bias if sigmoid routing (MiniMax)
    float *routing_bias = NULL;
    if (cfg.scoring_func == 1) {
        snprintf(name, sizeof(name), "model.layers.%d.%s.e_score_correction_bias", layer_idx, cfg.moe_prefix);
        routing_bias = get_tensor_ptr(wf, name);
    }

    uint32_t *sgw = NULL, *suw = NULL, *seg_w_local = NULL;
    uint16_t *sgs = NULL, *sus = NULL, *sgb = NULL, *sub = NULL, *seg_s_local = NULL, *seg_b_local = NULL;

    if (cfg.shared_intermediate > 0) {
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.weight", layer_idx);
        sgw = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.scales", layer_idx);
        sgs = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.biases", layer_idx);
        sgb = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.weight", layer_idx);
        suw = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.scales", layer_idx);
        sus = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.biases", layer_idx);
        sub = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.weight", layer_idx);
        seg_w_local = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.scales", layer_idx);
        seg_s_local = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.biases", layer_idx);
        seg_b_local = get_tensor_ptr(wf, name);
    }

    // Batch matmuls: routing gate + (optionally) shared expert
    if (gate_w && gate_s && gate_b) {
        if (cfg.gate_bits != cfg.bits) {
            // 8-bit routing gate: compute on CPU separately
            cpu_dequant_matvec_nbits(gate_w, gate_s, gate_b, h_post, gate_scores,
                                     cfg.num_experts, cfg.hidden_dim, cfg.gate_group_size, cfg.gate_bits);
            if (cfg.shared_intermediate > 0 && sgw && sgs && sgb && suw && sus && sub && seg_w_local && seg_s_local && seg_b_local) {
                BatchMatvecSpec moe_specs[3] = {
                    { sgw,    sgs,    sgb,    shared_gate,         (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 1 },
                    { suw,    sus,    sub,    shared_up,           (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 2 },
                    { seg_w_local,  seg_s_local,  seg_b_local,  &shared_gate_score,  1,                            cfg.hidden_dim, cfg.group_size, 3 },
                };
                fast_batch_matvec(h_post, cfg.hidden_dim, moe_specs, 3);
            }
        } else if (cfg.shared_intermediate > 0 && sgw && sgs && sgb && suw && sus && sub && seg_w_local && seg_s_local && seg_b_local) {
            BatchMatvecSpec moe_specs[4] = {
                { gate_w, gate_s, gate_b, gate_scores,        (uint32_t)cfg.num_experts,        cfg.hidden_dim, cfg.group_size, 0 },
                { sgw,    sgs,    sgb,    shared_gate,         (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 1 },
                { suw,    sus,    sub,    shared_up,           (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 2 },
                { seg_w_local,  seg_s_local,  seg_b_local,  &shared_gate_score,  1,                            cfg.hidden_dim, cfg.group_size, 3 },
            };
            fast_batch_matvec(h_post, cfg.hidden_dim, moe_specs, 4);
        } else {
            // No shared expert, same bits: just routing gate via GPU/fast
            BatchMatvecSpec moe_specs[1] = {
                { gate_w, gate_s, gate_b, gate_scores, (uint32_t)cfg.num_experts, cfg.hidden_dim, cfg.group_size, 0 },
            };
            fast_batch_matvec(h_post, cfg.hidden_dim, moe_specs, 1);
        }
    }

    // Route experts (softmax for Qwen, sigmoid+bias for MiniMax)
    int expert_indices[64];
    float expert_weights[64];
    cpu_route_experts(gate_scores, cfg.num_experts, K,
                      routing_bias, expert_indices, expert_weights);

    if (moe_dump) {
        fprintf(stderr, "[MOE-DUMP] routing: K=%d experts=[", K);
        for (int k = 0; k < K; k++) fprintf(stderr, "%d(%.4f)%s", expert_indices[k], expert_weights[k], k<K-1?",":"");
        fprintf(stderr, "]\n");
    }

    // ---- Routed expert computation ----
    float *moe_out = calloc(cfg.hidden_dim, sizeof(float));

    if (packed_fd >= 0) {
        float *expert_out = malloc(cfg.hidden_dim * sizeof(float));

        for (int k = 0; k < K; k++) {
            int eidx = expert_indices[k];
            off_t expert_offset; size_t esz;
            expert_offset_size(layer_idx, eidx, &expert_offset, &esz);

            if (g_metal && g_metal->buf_expert_data) {
                // GPU path: pread directly into Metal buffer, run gate+up+swiglu+down on GPU
                void *expert_buf_ptr = [g_metal->buf_expert_data contents];
                ssize_t nread = pread(packed_fd, expert_buf_ptr, esz, expert_offset);
                if (nread != (ssize_t)esz) {
                    fprintf(stderr, "WARNING: layer %d expert %d pread: %zd/%zu\n",
                            layer_idx, eidx, nread, esz);
                    continue;
                }

                gpu_expert_forward(g_metal, expert_buf_ptr, h_post, expert_out, 1 /*already in buffer*/);
            } else {
                // CPU fallback
                void *expert_data = malloc(esz);
                ssize_t nread = pread(packed_fd, expert_data, esz, expert_offset);
                if (nread != (ssize_t)esz) {
                    fprintf(stderr, "WARNING: layer %d expert %d pread: %zd/%zu\n",
                            layer_idx, eidx, nread, esz);
                    free(expert_data);
                    continue;
                }

                uint32_t *gw = (uint32_t *)expert_data;
                uint16_t *gs_p = (uint16_t *)((char *)expert_data + (g_use_2bit ? cfg.gate_s_off_2 : cfg.gate_s_off_4));
                uint16_t *gb_p = (uint16_t *)((char *)expert_data + (g_use_2bit ? cfg.gate_b_off_2 : cfg.gate_b_off_4));
                uint32_t *uw = (uint32_t *)((char *)expert_data + (g_use_2bit ? cfg.up_w_off_2 : cfg.up_w_off_4));
                uint16_t *us_p = (uint16_t *)((char *)expert_data + (g_use_2bit ? cfg.up_s_off_2 : cfg.up_s_off_4));
                uint16_t *ub_p = (uint16_t *)((char *)expert_data + (g_use_2bit ? cfg.up_b_off_2 : cfg.up_b_off_4));
                uint32_t *dw = (uint32_t *)((char *)expert_data + (g_use_2bit ? cfg.down_w_off_2 : cfg.down_w_off_4));
                uint16_t *ds_p = (uint16_t *)((char *)expert_data + (g_use_2bit ? cfg.down_s_off_2 : cfg.down_s_off_4));
                uint16_t *db_p = (uint16_t *)((char *)expert_data + (g_use_2bit ? cfg.down_b_off_2 : cfg.down_b_off_4));

                float *gate_proj_out = malloc(cfg.moe_intermediate * sizeof(float));
                float *up_proj_out = malloc(cfg.moe_intermediate * sizeof(float));
                float *act_out = malloc(cfg.moe_intermediate * sizeof(float));

                cpu_dequant_matvec(gw, gs_p, gb_p, h_post, gate_proj_out,
                                   cfg.moe_intermediate, cfg.hidden_dim, cfg.group_size);
                cpu_dequant_matvec(uw, us_p, ub_p, h_post, up_proj_out,
                                   cfg.moe_intermediate, cfg.hidden_dim, cfg.group_size);
                cpu_swiglu(gate_proj_out, up_proj_out, act_out, cfg.moe_intermediate);
                cpu_dequant_matvec(dw, ds_p, db_p, act_out, expert_out,
                                   cfg.hidden_dim, cfg.moe_intermediate, cfg.group_size);

                free(gate_proj_out);
                free(up_proj_out);
                free(act_out);
                free(expert_data);
            }

            // Accumulate weighted
            if (moe_dump) {
                fprintf(stderr, "[MOE-DUMP] expert[%d] out_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        eidx, vec_rms(expert_out, cfg.hidden_dim),
                        expert_out[0], expert_out[1], expert_out[2], expert_out[3], expert_out[4]);
            }
            cpu_vec_madd(moe_out, expert_out, expert_weights[k], cfg.hidden_dim);
        }

        free(expert_out);
    }

    // ---- Shared expert SwiGLU + combine ----
    float *shared_out = NULL;
    float *shared_act = NULL;

    if (cfg.shared_intermediate > 0 && shared_gate && shared_up) {
        shared_out = calloc(cfg.hidden_dim, sizeof(float));
        shared_act = calloc(cfg.shared_intermediate, sizeof(float));
        cpu_swiglu(shared_gate, shared_up, shared_act, cfg.shared_intermediate);

        if (moe_dump) {
            fprintf(stderr, "[MOE-DUMP] layer=%d h_post_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                    layer_idx, vec_rms(h_post, cfg.hidden_dim), h_post[0], h_post[1], h_post[2], h_post[3], h_post[4]);
            fprintf(stderr, "[MOE-DUMP] gate_proj_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                    vec_rms(shared_gate, cfg.shared_intermediate),
                    shared_gate[0], shared_gate[1], shared_gate[2], shared_gate[3], shared_gate[4]);
            fprintf(stderr, "[MOE-DUMP] up_proj_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                    vec_rms(shared_up, cfg.shared_intermediate),
                    shared_up[0], shared_up[1], shared_up[2], shared_up[3], shared_up[4]);
            fprintf(stderr, "[MOE-DUMP] swiglu_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                    vec_rms(shared_act, cfg.shared_intermediate),
                    shared_act[0], shared_act[1], shared_act[2], shared_act[3], shared_act[4]);
        }

        // shared_expert down_proj
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.weight", layer_idx);
        uint32_t *sdw = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.scales", layer_idx);
        uint16_t *sds = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.biases", layer_idx);
        uint16_t *sdb = get_tensor_ptr(wf, name);
        if (sdw && sds && sdb) {
            fast_dequant_matvec(sdw, sds, sdb, shared_act, shared_out, cfg.hidden_dim,
                                cfg.shared_intermediate, cfg.group_size);
        }

        // Shared expert gate (sigmoid)
        float shared_weight = cpu_sigmoid(shared_gate_score);
        for (int i = 0; i < cfg.hidden_dim; i++) shared_out[i] *= shared_weight;

        // Combine: hidden = h_mid + moe_out + shared_out
        for (int i = 0; i < cfg.hidden_dim; i++)
            hidden[i] = h_mid[i] + moe_out[i] + shared_out[i];

        if (moe_debug) {
            fprintf(stderr, "[MOE-DBG] layer=%d h_mid_rms=%.4f moe_rms=%.4f shared_rms=%.4f shared_gate=%.4f hidden_rms=%.4f\n",
                    layer_idx, vec_rms(h_mid, cfg.hidden_dim), vec_rms(moe_out, cfg.hidden_dim),
                    vec_rms(shared_out, cfg.hidden_dim), shared_weight,
                    vec_rms(hidden, cfg.hidden_dim));
        }
    } else {
        // No shared expert (MiniMax): hidden = h_mid + moe_out
        for (int i = 0; i < cfg.hidden_dim; i++)
            hidden[i] = h_mid[i] + moe_out[i];
    }

    free(h_post);
    free(h_mid);
    free(gate_scores);
    free(moe_out);
    free(shared_out);
    free(shared_gate);
    free(shared_up);
    free(shared_act);
}

// ============================================================================
// Embedding lookup (4-bit quantized)
// ============================================================================

void embed_lookup(WeightFile *wf, int token_id, float *out) {
    // Embedding: weight[vocab_size, hidden_dim/8] (U32), scales[vocab_size, groups], biases[vocab_size, groups]
    // For embedding lookup, we just need one row.
    // But the embedding is quantized: each row has hidden_dim/8 uint32 values (packed 4-bit)
    // plus scales and biases per group

    TensorInfo *w_info = get_tensor_info(wf, "model.embed_tokens.weight");
    TensorInfo *s_info = get_tensor_info(wf, "model.embed_tokens.scales");
    TensorInfo *b_info = get_tensor_info(wf, "model.embed_tokens.biases");

    if (!w_info || !s_info || !b_info) {
        fprintf(stderr, "ERROR: embedding tensors not found\n");
        memset(out, 0, cfg.hidden_dim * sizeof(float));
        return;
    }

    // w shape: [248320, 512] U32 -> each row has 512 uint32 = 4096 packed 4-bit values
    int packed_cols = w_info->shape[1];  // 512
    int num_groups = s_info->shape[1];   // 64

    uint32_t *W = (uint32_t *)((char *)wf->data + w_info->offset);
    uint16_t *S = (uint16_t *)((char *)wf->data + s_info->offset);
    uint16_t *B = (uint16_t *)((char *)wf->data + b_info->offset);

    const uint32_t *w_row = W + (size_t)token_id * packed_cols;
    const uint16_t *s_row = S + (size_t)token_id * num_groups;
    const uint16_t *b_row = B + (size_t)token_id * num_groups;

    int group_size = cfg.hidden_dim / num_groups;  // 4096/64 = 64
    int packed_per_group = group_size / 8;     // 8

    for (int g = 0; g < num_groups; g++) {
        float scale = bf16_to_f32(s_row[g]);
        float bias = bf16_to_f32(b_row[g]);

        for (int p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[g * packed_per_group + p];
            int base = g * group_size + p * 8;

            for (int n = 0; n < 8; n++) {
                uint32_t nibble = (packed >> (n * 4)) & 0xF;
                out[base + n] = (float)nibble * scale + bias;
            }
        }
    }
}

// ============================================================================
// LM head (logits projection)
// ============================================================================

void lm_head_forward(WeightFile *wf, const float *hidden, float *logits) {
    // lm_head: [hidden_dim=4096] -> [vocab_size=248320]
    // This is a HUGE matmul. For 248320 output dims, it will be slow on CPU.
    // Optimization: only compute top candidates

    TensorInfo *w_info = get_tensor_info(wf, "lm_head.weight");
    TensorInfo *s_info = get_tensor_info(wf, "lm_head.scales");
    TensorInfo *b_info = get_tensor_info(wf, "lm_head.biases");

    if (!w_info || !s_info || !b_info) {
        fprintf(stderr, "ERROR: lm_head tensors not found\n");
        return;
    }

    uint32_t *W = (uint32_t *)((char *)wf->data + w_info->offset);
    uint16_t *S = (uint16_t *)((char *)wf->data + s_info->offset);
    uint16_t *B = (uint16_t *)((char *)wf->data + b_info->offset);

    // Full matmul — use GPU if available (248320 output rows!)
    fast_dequant_matvec(W, S, B, hidden, logits, cfg.vocab_size, cfg.hidden_dim, cfg.group_size);
}

// ============================================================================
// Parallel I/O infrastructure for expert pread (from proven main.m pattern)
// ============================================================================

#define NUM_IO_THREADS 8  // 8 threads for K=8 experts (one per expert)

typedef struct {
    int fd;
    void *dst;
    off_t offset;
    size_t size;
    ssize_t result;
    const void *mmap_base;  // if non-NULL, memcpy from mmap instead of pread
    // LZ4 compression fields (set by caller when reading compressed experts)
    void *lz4_comp_buf;     // if non-NULL: pread into this, then LZ4 decompress into dst
    uint32_t lz4_comp_size; // compressed size to read from disk
} InferPreadTask;

typedef struct {
    InferPreadTask *tasks;
    int num_tasks;
    int thread_id;
} InferPreadThreadArg;

static void *infer_pread_thread_fn(void *arg) {
    InferPreadThreadArg *ta = (InferPreadThreadArg *)arg;
    for (int i = ta->thread_id; i < ta->num_tasks; i += NUM_IO_THREADS) {
        InferPreadTask *t = &ta->tasks[i];
        t->result = pread(t->fd, t->dst, t->size, t->offset);
    }
    return NULL;
}

// ============================================================================
// Persistent I/O Thread Pool — eliminates pthread_create/join per layer
// ============================================================================

typedef struct {
    pthread_t threads[NUM_IO_THREADS];
    pthread_mutex_t mutex;
    pthread_cond_t work_ready;
    pthread_cond_t work_done;
    InferPreadTask *tasks;
    int num_tasks;
    int tasks_completed;
    int generation;          // incremented each dispatch — workers wait for new gen
    int completed_generation;
    volatile int shutdown;
} IOThreadPool;

static IOThreadPool g_io_pool;
static int g_io_pool_initialized = 0;

static void *io_pool_worker(void *arg) {
    int tid = (int)(intptr_t)arg;
    int my_gen = 0;
    pthread_mutex_lock(&g_io_pool.mutex);
    while (1) {
        while (g_io_pool.generation == my_gen && !g_io_pool.shutdown)
            pthread_cond_wait(&g_io_pool.work_ready, &g_io_pool.mutex);
        if (g_io_pool.shutdown) break;
        my_gen = g_io_pool.generation;

        // Snapshot work for this generation
        int num_tasks = g_io_pool.num_tasks;
        InferPreadTask *tasks = g_io_pool.tasks;
        pthread_mutex_unlock(&g_io_pool.mutex);

        // Process assigned tasks (stride by thread count)
        for (int i = tid; i < num_tasks; i += NUM_IO_THREADS) {
            InferPreadTask *t = &tasks[i];
            if (t->lz4_comp_buf && t->lz4_comp_size > 0) {
                // LZ4 path: read compressed from SSD, decompress into dst
                ssize_t nr = pread(t->fd, t->lz4_comp_buf, t->lz4_comp_size, t->offset);
                if (nr == (ssize_t)t->lz4_comp_size) {
                    size_t dec = compression_decode_buffer(
                        t->dst, t->size, t->lz4_comp_buf, t->lz4_comp_size,
                        NULL, COMPRESSION_LZ4);
                    t->result = (ssize_t)dec;
                } else {
                    t->result = -1;
                }
            } else {
                t->result = pread(t->fd, t->dst, t->size, t->offset);
            }
        }

        pthread_mutex_lock(&g_io_pool.mutex);
        g_io_pool.tasks_completed++;
        if (g_io_pool.tasks_completed == NUM_IO_THREADS) {
            g_io_pool.completed_generation = my_gen;
            pthread_cond_signal(&g_io_pool.work_done);
        }
    }
    pthread_mutex_unlock(&g_io_pool.mutex);
    return NULL;
}

void io_pool_init(void) {
    if (g_io_pool_initialized) return;
    pthread_mutex_init(&g_io_pool.mutex, NULL);
    pthread_cond_init(&g_io_pool.work_ready, NULL);
    pthread_cond_init(&g_io_pool.work_done, NULL);
    g_io_pool.shutdown = 0;
    g_io_pool.generation = 0;
    g_io_pool.completed_generation = 0;
    g_io_pool.tasks = NULL;
    for (int i = 0; i < NUM_IO_THREADS; i++)
        pthread_create(&g_io_pool.threads[i], NULL, io_pool_worker, (void*)(intptr_t)i);
    g_io_pool_initialized = 1;
}

static dispatch_queue_t g_io_gcd_queue = NULL;

// Async start — returns generation number for later wait
static int io_pool_start(InferPreadTask *tasks, int num_tasks) {
    if (num_tasks == 0) return 0;
    pthread_mutex_lock(&g_io_pool.mutex);
    g_io_pool.tasks = tasks;
    g_io_pool.num_tasks = num_tasks;
    g_io_pool.tasks_completed = 0;
    g_io_pool.generation++;
    int gen = g_io_pool.generation;
    pthread_cond_broadcast(&g_io_pool.work_ready);
    pthread_mutex_unlock(&g_io_pool.mutex);
    return gen;
}

// Wait for a specific generation to complete
static void io_pool_wait_generation(int target_gen) {
    if (target_gen <= 0) return;
    pthread_mutex_lock(&g_io_pool.mutex);
    while (g_io_pool.completed_generation < target_gen) {
        pthread_cond_wait(&g_io_pool.work_done, &g_io_pool.mutex);
    }
    pthread_mutex_unlock(&g_io_pool.mutex);
}

// Synchronous dispatch — start + wait
static void io_pool_dispatch(InferPreadTask *tasks, int num_tasks) {
    if (num_tasks == 0) return;
    int my_gen = io_pool_start(tasks, num_tasks);
    io_pool_wait_generation(my_gen);
}

// ---- Async expert pread pipeline ----
// Uses GCD dispatch_group for truly async start+wait (no generation conflicts
// with the persistent io_pool). When cache_io_split > 1, each expert blob is
// split into N page-aligned chunks for parallel SSD reads.
#define MAX_CACHE_IO_SPLIT 8

static inline int active_cache_io_split(size_t esz) {
    int chunks = g_cache_io_split;
    if (chunks < 1) chunks = 1;
    if (chunks > MAX_CACHE_IO_SPLIT) chunks = MAX_CACHE_IO_SPLIT;

    // Expert blobs are page-cache-backed. Keep chunk boundaries page aligned
    // so fanout mode still matches the underlying VM layout.
    const size_t page_bytes = 16 * 1024;
    if (esz == 0 || (esz % page_bytes) != 0) return 1;

    size_t pages = esz / page_bytes;
    if ((size_t)chunks > pages) chunks = (int)pages;
    if (chunks < 1) chunks = 1;
    return chunks;
}

typedef struct {
    InferPreadTask tasks[MAX_K * MAX_CACHE_IO_SPLIT];
    int num_tasks;
    int num_experts;
    int chunks_per_expert;
    int valid[MAX_K];
    dispatch_group_t group;
    int active;
} AsyncPreadState;
static AsyncPreadState g_async_pread = {0};

static void async_pread_start(int packed_fd, int *expert_indices, int K,
                               id<MTLBuffer> __strong *dst_bufs, const void *mmap_base,
                               int layer_idx) {
    (void)mmap_base;
    size_t esz = active_expert_size();
    int chunks = active_cache_io_split(esz);
    const size_t page_bytes = 16 * 1024;

    g_async_pread.num_experts = K;
    g_async_pread.chunks_per_expert = chunks;
    g_async_pread.num_tasks = K * chunks;
    g_async_pread.active = 1;
    if (!g_async_pread.group) g_async_pread.group = dispatch_group_create();

    for (int k = 0; k < K; k++) {
        // Per-expert offset and size (tiered: variable, uniform: computed from index)
        size_t this_esz;
        off_t this_offset;
        if (g_use_tiered && g_tiered_manifest) {
            TieredExpertInfo *ti = &TIERED(layer_idx, expert_indices[k]);
            this_esz = ti->size;
            this_offset = (off_t)ti->offset;
        } else {
            this_esz = esz;
            this_offset = (off_t)expert_indices[k] * esz;
        }

        size_t total_pages = (chunks > 1) ? (this_esz / page_bytes) : 0;
        char *dst_base = (char *)[dst_bufs[k] contents];
        size_t page_cursor = 0;
        for (int c = 0; c < chunks; c++) {
            size_t chunk_off = 0;
            size_t chunk_sz = this_esz;
            if (chunks > 1) {
                size_t pages_this_chunk = total_pages / (size_t)chunks;
                if ((size_t)c < (total_pages % (size_t)chunks)) pages_this_chunk++;
                chunk_off = page_cursor * page_bytes;
                chunk_sz = pages_this_chunk * page_bytes;
                page_cursor += pages_this_chunk;
            }

            int task_idx = k * chunks + c;
            g_async_pread.tasks[task_idx].fd = packed_fd;
            g_async_pread.tasks[task_idx].dst = dst_base + chunk_off;
            g_async_pread.tasks[task_idx].offset = this_offset + (off_t)chunk_off;
            g_async_pread.tasks[task_idx].size = chunk_sz;
            g_async_pread.tasks[task_idx].result = 0;
            g_async_pread.tasks[task_idx].mmap_base = NULL;
            g_async_pread.tasks[task_idx].lz4_comp_buf = NULL;
            g_async_pread.tasks[task_idx].lz4_comp_size = 0;
        }
    }

    // Fire off parallel preads on GCD — dispatch_group guarantees all blocks
    // complete before dispatch_group_wait returns (no generation counter race).
    static dispatch_queue_t io_q = NULL;
    if (!io_q) io_q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
    int total_tasks = g_async_pread.num_tasks;
    for (int i = 0; i < total_tasks; i++) {
        InferPreadTask *t = &g_async_pread.tasks[i];
        dispatch_group_async(g_async_pread.group, io_q, ^{
            t->result = pread(t->fd, t->dst, t->size, t->offset);
        });
    }
}

static void async_pread_wait(void) {
    if (!g_async_pread.active) return;
    dispatch_group_wait(g_async_pread.group, DISPATCH_TIME_FOREVER);
    // Validate each chunk against its OWN expected size (not uniform esz).
    // Critical for tiered mode where cold 2-bit experts are smaller than hot 4-bit.
    for (int k = 0; k < g_async_pread.num_experts; k++) {
        int ok = 1;
        for (int c = 0; c < g_async_pread.chunks_per_expert; c++) {
            int task_idx = k * g_async_pread.chunks_per_expert + c;
            if (g_async_pread.tasks[task_idx].result != (ssize_t)g_async_pread.tasks[task_idx].size) {
                ok = 0;
                break;
            }
        }
        g_async_pread.valid[k] = ok;
    }
    g_async_pread.active = 0;
}

static void io_pool_shutdown(void) {
    if (!g_io_pool_initialized) return;
    pthread_mutex_lock(&g_io_pool.mutex);
    g_io_pool.shutdown = 1;
    pthread_cond_broadcast(&g_io_pool.work_ready);
    pthread_mutex_unlock(&g_io_pool.mutex);
    for (int i = 0; i < NUM_IO_THREADS; i++)
        pthread_join(g_io_pool.threads[i], NULL);
    pthread_mutex_destroy(&g_io_pool.mutex);
    pthread_cond_destroy(&g_io_pool.work_ready);
    pthread_cond_destroy(&g_io_pool.work_done);
    g_io_pool_initialized = 0;
}

// Parallel pread of K experts into Metal buffers using pthreads.
// Returns number of successfully loaded experts, sets valid[] flags.
static int parallel_pread_experts(
    int packed_fd,
    int *expert_indices,
    int K,
    int *valid,  // [MAX_K] output: 1 if expert loaded successfully
    const void *mmap_base,  // mmap'd layer file (NULL to use pread)
    int layer_idx  // needed for tiered manifest lookup
) {
    size_t esz = active_expert_size();
    InferPreadTask tasks[MAX_K];
    for (int k = 0; k < K; k++) {
        size_t this_esz;
        off_t this_offset;
        if (g_use_tiered && g_tiered_manifest) {
            TieredExpertInfo *ti = &TIERED(layer_idx, expert_indices[k]);
            this_esz = ti->size;
            this_offset = (off_t)ti->offset;
        } else {
            this_esz = esz;
            this_offset = (off_t)expert_indices[k] * esz;
        }
        tasks[k].fd = packed_fd;
        tasks[k].dst = [g_metal->buf_multi_expert_data[k] contents];
        tasks[k].offset = this_offset;
        tasks[k].size = this_esz;
        tasks[k].result = 0;
        tasks[k].mmap_base = mmap_base;
    }

    io_pool_dispatch(tasks, K);

    int loaded = 0;
    for (int k = 0; k < K; k++) {
        valid[k] = (tasks[k].result == (ssize_t)tasks[k].size);
        if (valid[k]) loaded++;
        else {
            fprintf(stderr, "WARNING: expert %d pread: %zd/%zu\n",
                    expert_indices[k], tasks[k].result, tasks[k].size);
        }
    }
    return loaded;
}

// ============================================================================
// Parallel pread into explicit buffer set (for double buffering).
// Same as parallel_pread_experts but reads into caller-specified MTLBuffers.
// ============================================================================
static int parallel_pread_experts_into(
    int packed_fd,
    int *expert_indices,
    int K,
    id<MTLBuffer> __strong *dst_bufs,  // target Metal buffers (set A or B)
    int *valid,  // [MAX_K] output: 1 if expert loaded successfully
    int layer_idx  // needed for tiered manifest lookup
) {
    size_t esz = active_expert_size();
    InferPreadTask tasks[MAX_K];
    for (int k = 0; k < K; k++) {
        size_t this_esz;
        off_t this_offset;
        if (g_use_tiered && g_tiered_manifest) {
            TieredExpertInfo *ti = &TIERED(layer_idx, expert_indices[k]);
            this_esz = ti->size;
            this_offset = (off_t)ti->offset;
        } else {
            this_esz = esz;
            this_offset = (off_t)expert_indices[k] * esz;
        }
        tasks[k].fd = packed_fd;
        tasks[k].dst = [dst_bufs[k] contents];
        tasks[k].offset = this_offset;
        tasks[k].size = this_esz;
        tasks[k].result = 0;
    }

    io_pool_dispatch(tasks, K);

    int loaded = 0;
    for (int k = 0; k < K; k++) {
        valid[k] = (tasks[k].result == (ssize_t)tasks[k].size);
        if (valid[k]) loaded++;
        else {
            fprintf(stderr, "WARNING: expert %d pread: %zd/%zu\n",
                    expert_indices[k], tasks[k].result, tasks[k].size);
        }
    }
    return loaded;
}

// ============================================================================
// Expert LRU Cache: keeps recently-used expert Metal buffers in GPU memory.
//
// Key: (layer_idx, expert_idx) -> Metal buffer containing 7.08MB expert data.
// On cache HIT:  skip pread entirely, use the cached Metal buffer for GPU dispatch.
// On cache MISS: pread into a new/evicted Metal buffer, insert into cache.
// LRU eviction:  when cache is full, evict the least recently used entry.
//
// Memory budget: 2000 entries * 7.08MB = 14.2GB. With 5.5GB non-expert weights
// + 14.2GB cache = 19.7GB total. Fits in 48GB with room for OS.
//
// Unlike Python/MLX where LRU caching caused Metal heap pressure and slower
// mx.eval(), here Metal buffers ARE the cache -- no conversion overhead.
// ============================================================================

typedef struct {
    int layer_idx;
    int expert_idx;
    id<MTLBuffer> buffer;    // Metal buffer holding cfg.expert_size_4bit bytes
    uint64_t last_used;      // monotonic counter for LRU ordering
} ExpertCacheEntry;

typedef struct {
    ExpertCacheEntry *entries;
    int max_entries;
    int num_entries;
    int used_entries;
    int *entry_idx;  // flattened [num_layers * num_experts], -1 = not cached
    uint64_t access_counter; // monotonic, incremented on every access
    id<MTLDevice> device;    // for allocating new Metal buffers
    // Stats
    uint64_t hits;
    uint64_t misses;
} ExpertLRUCache;

static ExpertLRUCache *g_expert_cache = NULL;

// Speculative early routing stats
static uint64_t g_spec_route_attempts = 0;   // total speculative routing attempts
static uint64_t g_spec_route_hits = 0;        // correctly predicted experts (found in cache at real routing time)
static uint64_t g_spec_route_preloads = 0;    // async preloads initiated (cache misses at speculation time)

// ---- Temporal prediction pipeline ----
// Stores previous token's expert routing per layer. On the next token,
// predicted experts are preloaded into buf_multi_expert_data_B during CMD1_wait
// idle time. After routing, hits use buf_B, misses sync-pread into buf_A.
// Different from previous failed speculative attempts:
//   - Loads into scratch buffers (no cache pollution)
//   - Uses CMD1_wait idle time (no additional CPU cost)
//   - Only sync-preads misses (not all K experts)
static int g_pred_valid = 0;                       // 1 after first token completes (predictions available)
// g_pred_enabled, g_pred_hits, g_pred_misses, g_pred_layers declared near timing (line ~163)

static ExpertLRUCache *expert_cache_new(id<MTLDevice> device, int max_entries) {
    ExpertLRUCache *cache = calloc(1, sizeof(ExpertLRUCache));
    cache->entries = calloc(max_entries, sizeof(ExpertCacheEntry));
    cache->entry_idx = malloc(cfg.num_layers * cfg.num_experts * sizeof(int));
    cache->max_entries = max_entries;
    cache->num_entries = 0;
    cache->used_entries = 0;
    cache->access_counter = 0;
    cache->device = device;
    cache->hits = 0;
    cache->misses = 0;
    for (int l = 0; l < cfg.num_layers; l++) {
        for (int e = 0; e < cfg.num_experts; e++) {
            cache->entry_idx[(l) * cfg.num_experts + (e)] = -1;
        }
    }
    // Pre-allocate ALL Metal buffers at startup (avoids allocation overhead at runtime)
    size_t esz = active_expert_size();
    double t_prealloc = now_ms();
    for (int i = 0; i < max_entries; i++) {
        cache->entries[i].buffer = [device newBufferWithLength:esz
                                                      options:MTLResourceStorageModeShared];
        cache->entries[i].layer_idx = -1;
        cache->entries[i].expert_idx = -1;
        cache->entries[i].last_used = 0;
        if (!cache->entries[i].buffer) {
            fprintf(stderr, "WARNING: expert_cache: pre-alloc failed at entry %d\n", i);
            max_entries = i;
            cache->max_entries = i;
            break;
        }
    }
    cache->num_entries = max_entries; // All slots pre-allocated (but empty keys)
    printf("[expert_cache] Initialized: max_entries=%d (%.1f GB budget), pre-alloc %.0f ms\n",
           max_entries, (double)max_entries * esz / 1e9, now_ms() - t_prealloc);
    return cache;
}

static void expert_cache_free(ExpertLRUCache *cache) {
    if (!cache) return;
    printf("[expert_cache] Final stats: %llu hits, %llu misses (%.1f%% hit rate)\n",
           cache->hits, cache->misses,
           (cache->hits + cache->misses) > 0
               ? 100.0 * cache->hits / (cache->hits + cache->misses) : 0.0);
    // Metal buffers released by ARC when entries are freed
    free(cache->entry_idx);
    free(cache->entries);
    free(cache);
}

// Lookup: returns the cached Metal buffer if found, otherwise NULL.
// On hit, updates the LRU timestamp.
static id<MTLBuffer> expert_cache_lookup(ExpertLRUCache *cache, int layer_idx, int expert_idx) {
    int idx = cache->entry_idx[(layer_idx) * cfg.num_experts + (expert_idx)];
    if (idx >= 0) {
        cache->entries[idx].last_used = ++cache->access_counter;
        cache->hits++;
        cache_telemetry_touch(layer_idx, expert_idx);
        return cache->entries[idx].buffer;
    }
    cache->misses++;
    cache_telemetry_miss(layer_idx, expert_idx);
    return nil;
}

// Insert: adds a new entry. If the cache is full, evicts the LRU entry.
// Returns the Metal buffer to pread into (either newly allocated or evicted+reused).
static id<MTLBuffer> expert_cache_insert(ExpertLRUCache *cache, int layer_idx, int expert_idx) {
    id<MTLBuffer> buf = nil;

    int existing = cache->entry_idx[(layer_idx) * cfg.num_experts + (expert_idx)];
    if (existing >= 0) {
        cache->entries[existing].last_used = ++cache->access_counter;
        return cache->entries[existing].buffer;
    }

    // Find a slot: first try an unused slot (layer_idx == -1), then LRU evict
    int target = -1;
    if (cache->used_entries < cache->num_entries) {
        target = cache->used_entries++;
    }
    if (target >= 0) {
        // Unused pre-allocated slot
        buf = cache->entries[target].buffer;
        cache->entries[target].layer_idx = layer_idx;
        cache->entries[target].expert_idx = expert_idx;
        cache->entries[target].last_used = ++cache->access_counter;
        cache->entry_idx[(layer_idx) * cfg.num_experts + (expert_idx)] = target;
        return buf;
    }

    // Cache full: find LRU entry (smallest last_used)
    int lru_idx = 0;
    uint64_t min_used = cache->entries[0].last_used;
    for (int i = 1; i < cache->num_entries; i++) {
        if (cache->entries[i].last_used < min_used) {
            min_used = cache->entries[i].last_used;
            lru_idx = i;
        }
    }

    // Reuse the evicted entry's Metal buffer (same size, no realloc needed)
    int old_layer = cache->entries[lru_idx].layer_idx;
    int old_expert = cache->entries[lru_idx].expert_idx;
    cache_telemetry_evict(old_layer, old_expert);
    if (old_layer >= 0 && old_expert >= 0) {
        cache->entry_idx[(old_layer) * cfg.num_experts + (old_expert)] = -1;
    }
    buf = cache->entries[lru_idx].buffer;
    cache->entries[lru_idx].layer_idx = layer_idx;
    cache->entries[lru_idx].expert_idx = expert_idx;
    cache->entries[lru_idx].last_used = ++cache->access_counter;
    cache->entry_idx[(layer_idx) * cfg.num_experts + (expert_idx)] = lru_idx;
    return buf;
}

// ============================================================================
// Malloc-based expert frequency cache.
// Stores expert data in regular malloc'd memory (not Metal buffers) to avoid
// GPU memory pressure. On hit, memcpy to Metal scratch buffer. Much larger
// capacity than Metal buffer LRU cache at the cost of one memcpy per hit.
// ============================================================================

typedef struct {
    void **data;           // [max_entries] page-aligned malloc'd cfg.expert_size_4bit buffers
    id<MTLBuffer> __strong *metal_bufs;  // [max_entries] zero-copy Metal buffer wrappers
    int *layer_idx;        // [max_entries] layer index for each entry
    int *expert_idx;       // [max_entries] expert index for each entry
    uint64_t *last_used;   // [max_entries] monotonic counter for LRU
    int max_entries;
    int num_entries;
    int used_entries;
    int *entry_idx;  // flattened [num_layers * num_experts], -1 = not cached
    uint64_t access_counter;
    uint64_t hits;
    uint64_t misses;
} MallocExpertCache;

static MallocExpertCache *g_malloc_cache = NULL;

static MallocExpertCache *malloc_cache_init(int max_entries, id<MTLDevice> device) {
    MallocExpertCache *cache = calloc(1, sizeof(MallocExpertCache));
    cache->data = calloc(max_entries, sizeof(void *));
    cache->metal_bufs = (__strong id<MTLBuffer> *)calloc(max_entries, sizeof(id<MTLBuffer>));
    cache->layer_idx = calloc(max_entries, sizeof(int));
    cache->expert_idx = calloc(max_entries, sizeof(int));
    cache->last_used = calloc(max_entries, sizeof(uint64_t));
    cache->entry_idx = malloc(cfg.num_layers * cfg.num_experts * sizeof(int));
    cache->max_entries = max_entries;
    cache->num_entries = 0;
    cache->used_entries = 0;
    cache->access_counter = 0;
    cache->hits = 0;
    cache->misses = 0;
    for (int l = 0; l < cfg.num_layers; l++) {
        for (int e = 0; e < cfg.num_experts; e++) {
            cache->entry_idx[(l) * cfg.num_experts + (e)] = -1;
        }
    }

    size_t esz = active_expert_size();
    printf("[malloc_cache] Initializing: %d entries (%.1f GB) with zero-copy Metal wrappers\n",
           max_entries, (double)max_entries * esz / 1e9);
    double t_start = now_ms();

    size_t page_size = (size_t)getpagesize();
    // Round expert size up to page boundary for newBufferWithBytesNoCopy
    size_t aligned_size = (esz + page_size - 1) & ~(page_size - 1);

    for (int i = 0; i < max_entries; i++) {
        // Page-aligned allocation for zero-copy Metal buffer
        void *buf = NULL;
        if (posix_memalign(&buf, page_size, aligned_size) != 0 || !buf) {
            fprintf(stderr, "WARNING: malloc_cache: alloc failed at entry %d\n", i);
            max_entries = i;
            cache->max_entries = i;
            break;
        }
        memset(buf, 0, aligned_size);
        cache->data[i] = buf;

        // Create zero-copy Metal buffer wrapping the malloc'd memory
        // nil deallocator = Metal doesn't free the memory
        cache->metal_bufs[i] = [device newBufferWithBytesNoCopy:buf
                                                         length:aligned_size
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        cache->layer_idx[i] = -1;
        cache->expert_idx[i] = -1;
        cache->last_used[i] = 0;
    }
    cache->num_entries = max_entries;

    printf("[malloc_cache] Pre-allocated %d entries in %.0f ms\n",
           max_entries, now_ms() - t_start);
    return cache;
}

// Lookup: returns Metal buffer wrapping cached data, or nil. Zero-copy dispatch.
static id<MTLBuffer> malloc_cache_lookup(MallocExpertCache *cache, int layer, int expert) {
    int idx = cache->entry_idx[(layer) * cfg.num_experts + (expert)];
    if (idx >= 0) {
        cache->last_used[idx] = ++cache->access_counter;
        cache->hits++;
        cache_telemetry_touch(layer, expert);
        return cache->metal_bufs[idx];
    }
    cache->misses++;
    cache_telemetry_miss(layer, expert);
    return nil;
}

// Insert: evict LRU if needed, return entry index for pread target.
// Returns the Metal buffer for this entry (caller should pread into cache->data[idx]).
static id<MTLBuffer> malloc_cache_insert(MallocExpertCache *cache, int layer, int expert, int *out_idx) {
    int existing = cache->entry_idx[(layer) * cfg.num_experts + (expert)];
    if (existing >= 0) {
        cache->last_used[existing] = ++cache->access_counter;
        if (out_idx) *out_idx = existing;
        return cache->metal_bufs[existing];
    }

    // Find a free slot (layer_idx == -1) or evict LRU
    int target = -1;
    if (cache->used_entries < cache->num_entries) {
        target = cache->used_entries++;
    }

    if (target < 0) {
        // Cache full: evict entry with smallest last_used
        target = 0;
        uint64_t min_used = cache->last_used[0];
        for (int i = 1; i < cache->num_entries; i++) {
            if (cache->last_used[i] < min_used) {
                min_used = cache->last_used[i];
                target = i;
            }
        }
        cache_telemetry_evict(cache->layer_idx[target], cache->expert_idx[target]);
        if (cache->layer_idx[target] >= 0 && cache->expert_idx[target] >= 0) {
            cache->entry_idx[(cache->layer_idx[target]) * cfg.num_experts + (cache->expert_idx[target])] = -1;
        }
    }

    cache->layer_idx[target] = layer;
    cache->expert_idx[target] = expert;
    cache->last_used[target] = ++cache->access_counter;
    cache->entry_idx[(layer) * cfg.num_experts + (expert)] = target;
    if (out_idx) *out_idx = target;
    return cache->metal_bufs[target];
}

static void malloc_cache_free(MallocExpertCache *cache) {
    if (!cache) return;
    printf("[malloc_cache] Final stats: %llu hits, %llu misses (%.1f%% hit rate)\n",
           cache->hits, cache->misses,
           (cache->hits + cache->misses) > 0
               ? 100.0 * cache->hits / (cache->hits + cache->misses) : 0.0);
    for (int i = 0; i < cache->num_entries; i++) {
        cache->metal_bufs[i] = nil;  // release Metal buffer wrapper
        free(cache->data[i]);
    }
    free(cache->data);
    free(cache->metal_bufs);
    free(cache->entry_idx);
    free(cache->layer_idx);
    free(cache->expert_idx);
    free(cache->last_used);
    free(cache);
}

// ============================================================================
// Background prefetch thread for double-buffered expert I/O (from main.m).
// Runs pread on a background thread while main thread does GPU compute.
// Uses pure C I/O plan to avoid ARC issues across threads.
// ============================================================================

typedef struct {
    void *dst[MAX_K];       // raw pointers from [buf contents] (no ARC)
    off_t offset[MAX_K];    // file offsets per expert
    size_t size[MAX_K];     // bytes to read per expert (may vary in tiered mode)
    int K;                  // number of experts
    int fd;                 // file descriptor for this layer
    int valid[MAX_K];       // output: 1 if pread succeeded
    int loaded;             // output: count of successfully loaded experts
} InferIOPlan;

typedef struct {
    InferIOPlan plan;       // pre-built I/O plan (pure C, no ARC)
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int start;              // signal: set to 1 to start prefetch
    int done;               // signal: set to 1 when prefetch complete
    int shutdown;           // signal: set to 1 to exit thread
} InferPrefetchCtx;

static void *infer_prefetch_thread_fn(void *arg) {
    InferPrefetchCtx *pf = (InferPrefetchCtx *)arg;

    while (1) {
        pthread_mutex_lock(&pf->mutex);
        while (!pf->start && !pf->shutdown) {
            pthread_cond_wait(&pf->cond, &pf->mutex);
        }
        if (pf->shutdown) {
            pthread_mutex_unlock(&pf->mutex);
            break;
        }
        pf->start = 0;
        pthread_mutex_unlock(&pf->mutex);

        // Execute parallel pread (pure C, no ARC objects)
        InferIOPlan *plan = &pf->plan;
        InferPreadTask tasks[MAX_K];
        for (int k = 0; k < plan->K; k++) {
            tasks[k].fd = plan->fd;
            tasks[k].dst = plan->dst[k];
            tasks[k].offset = plan->offset[k];
            tasks[k].size = plan->size[k];
            tasks[k].result = 0;
        }

        io_pool_dispatch(tasks, plan->K);

        plan->loaded = 0;
        for (int k = 0; k < plan->K; k++) {
            plan->valid[k] = (tasks[k].result == (ssize_t)plan->size[k]);
            if (plan->valid[k]) plan->loaded++;
        }

        // Signal completion
        pthread_mutex_lock(&pf->mutex);
        pf->done = 1;
        pthread_cond_signal(&pf->cond);
        pthread_mutex_unlock(&pf->mutex);
    }

    return NULL;
}

// Build I/O plan on main thread (ARC-safe: extracts void* from id<MTLBuffer>),
// then signal background prefetch thread.
static void infer_prefetch_start(InferPrefetchCtx *pf, int packed_fd,
                                  int *expert_indices, int K,
                                  id<MTLBuffer> __strong *dst_bufs,
                                  int layer_idx) {
    pthread_mutex_lock(&pf->mutex);
    InferIOPlan *plan = &pf->plan;
    plan->fd = packed_fd;
    plan->K = K;
    for (int k = 0; k < K; k++) {
        off_t eoff; size_t esz;
        expert_offset_size(layer_idx, expert_indices[k], &eoff, &esz);
        plan->dst[k] = [dst_bufs[k] contents];
        plan->offset[k] = eoff;
        plan->size[k] = esz;
        plan->valid[k] = 0;
    }
    plan->loaded = 0;
    pf->done = 0;
    pf->start = 1;
    pthread_cond_signal(&pf->cond);
    pthread_mutex_unlock(&pf->mutex);
}

// Wait for background prefetch to complete. Returns number of loaded experts.
// Copies valid[] flags into caller's array.
static int infer_prefetch_wait(InferPrefetchCtx *pf, int *valid_out, int K) {
    pthread_mutex_lock(&pf->mutex);
    while (!pf->done) {
        pthread_cond_wait(&pf->cond, &pf->mutex);
    }
    int loaded = pf->plan.loaded;
    for (int k = 0; k < K; k++) {
        valid_out[k] = pf->plan.valid[k];
    }
    pthread_mutex_unlock(&pf->mutex);
    return loaded;
}

static InferPrefetchCtx *g_prefetch = NULL;
static pthread_t g_prefetch_tid;

static void infer_prefetch_init(void) {
    if (g_prefetch) return;
    g_prefetch = calloc(1, sizeof(InferPrefetchCtx));
    pthread_mutex_init(&g_prefetch->mutex, NULL);
    pthread_cond_init(&g_prefetch->cond, NULL);
    g_prefetch->shutdown = 0;
    pthread_create(&g_prefetch_tid, NULL, infer_prefetch_thread_fn, g_prefetch);
}

static void infer_prefetch_shutdown(void) {
    if (!g_prefetch) return;
    pthread_mutex_lock(&g_prefetch->mutex);
    g_prefetch->shutdown = 1;
    pthread_cond_signal(&g_prefetch->cond);
    pthread_mutex_unlock(&g_prefetch->mutex);
    pthread_join(g_prefetch_tid, NULL);
    pthread_mutex_destroy(&g_prefetch->mutex);
    pthread_cond_destroy(&g_prefetch->cond);
    free(g_prefetch);
    g_prefetch = NULL;
}

// ============================================================================
// Per-layer weight pointer cache — built once, eliminates 40+ snprintf+lookup
// per layer per token. With 40 layers and 15 tokens = 24,000 lookups saved.
// ============================================================================

typedef struct {
    // Input/post-attention layer norms
    uint16_t *input_norm_w;
    uint16_t *post_attn_norm_w;

    // Full attention weights (non-NULL only for full attention layers)
    uint32_t *q_w; uint16_t *q_s, *q_b;
    uint32_t *k_w; uint16_t *k_s, *k_b;
    uint32_t *v_w; uint16_t *v_s, *v_b;
    uint32_t *o_w; uint16_t *o_s, *o_b;
    uint16_t *q_norm_w, *k_norm_w;

    // Linear attention weights (non-NULL only for linear attention layers)
    uint32_t *qkv_w; uint16_t *qkv_s, *qkv_b;
    uint32_t *z_w;   uint16_t *z_s, *z_b;
    uint32_t *b_w;   uint16_t *b_s, *b_b;
    uint32_t *a_w;   uint16_t *a_s, *a_b;
    uint16_t *conv1d_w;
    float *A_log;
    uint16_t *dt_bias;
    uint16_t *gated_norm_w;
    uint32_t *out_proj_w; uint16_t *out_proj_s, *out_proj_b;

    // MoE routing + shared expert weights
    uint32_t *gate_w; uint16_t *gate_s, *gate_b;
    float *routing_bias;                         // e_score_correction_bias (MiniMax, NULL for Qwen)
    uint32_t *sg_w;   uint16_t *sg_s, *sg_b;   // shared gate_proj
    uint32_t *su_w;   uint16_t *su_s, *su_b;   // shared up_proj
    uint32_t *sd_w;   uint16_t *sd_s, *sd_b;   // shared down_proj
    uint32_t *seg_w;  uint16_t *seg_s, *seg_b; // shared_expert_gate
} LayerWeightCache;

static LayerWeightCache *layer_cache = NULL;
static int layer_cache_built = 0;

// Allocate all dynamic tracking arrays (must be called after load_model_config)
void alloc_tracking_arrays(void) {
    int nl = cfg.num_layers;
    int ne = cfg.num_experts;
    int seen_bytes_per_layer = (ne + 7) / 8;

    g_expert_freq            = calloc(nl * ne, sizeof(int));
    g_expert_seen            = calloc(nl * seen_bytes_per_layer, sizeof(uint8_t));
    g_lz4_index              = calloc(nl, sizeof(void *));
    g_cache_seen             = calloc(nl * ne, sizeof(uint8_t));
    g_cache_last_touch_token = calloc(nl * ne, sizeof(uint64_t));
    g_cache_last_evict_token = calloc(nl * ne, sizeof(uint64_t));
    g_pred_experts           = calloc(nl * MAX_K, sizeof(int));
    g_pred_count             = calloc(nl, sizeof(int));
    layer_cache              = calloc(nl, sizeof(LayerWeightCache));
}

static void build_layer_cache(WeightFile *wf) {
    if (layer_cache_built) return;
    char name[256];

    for (int i = 0; i < cfg.num_layers; i++) {
        LayerWeightCache *lc = &layer_cache[i];
        int is_full = cfg.is_full_attn[i];

        // Norms
        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", i);
        lc->input_norm_w = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", i);
        lc->post_attn_norm_w = get_tensor_ptr(wf, name);

        if (is_full) {
            // Full attention
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", i);
            lc->q_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.scales", i);
            lc->q_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.biases", i);
            lc->q_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", i);
            lc->k_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.scales", i);
            lc->k_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.biases", i);
            lc->k_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", i);
            lc->v_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.scales", i);
            lc->v_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.biases", i);
            lc->v_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", i);
            lc->o_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.scales", i);
            lc->o_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.biases", i);
            lc->o_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", i);
            lc->q_norm_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", i);
            lc->k_norm_w = get_tensor_ptr(wf, name);
        } else {
            // Linear attention
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.weight", i);
            lc->qkv_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.scales", i);
            lc->qkv_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.biases", i);
            lc->qkv_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.weight", i);
            lc->z_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.scales", i);
            lc->z_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.biases", i);
            lc->z_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.weight", i);
            lc->b_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.scales", i);
            lc->b_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.biases", i);
            lc->b_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.weight", i);
            lc->a_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.scales", i);
            lc->a_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.biases", i);
            lc->a_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.conv1d.weight", i);
            lc->conv1d_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.A_log", i);
            lc->A_log = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.dt_bias", i);
            lc->dt_bias = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.norm.weight", i);
            lc->gated_norm_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.weight", i);
            lc->out_proj_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.scales", i);
            lc->out_proj_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.biases", i);
            lc->out_proj_b = get_tensor_ptr(wf, name);
        }

        // MoE routing gate (uses cfg.moe_prefix: "mlp" for Qwen, "block_sparse_moe" for MiniMax)
        snprintf(name, sizeof(name), "model.layers.%d.%s.gate.weight", i, cfg.moe_prefix);
        lc->gate_w = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.%s.gate.scales", i, cfg.moe_prefix);
        lc->gate_s = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.%s.gate.biases", i, cfg.moe_prefix);
        lc->gate_b = get_tensor_ptr(wf, name);

        // Routing bias (MiniMax e_score_correction_bias, NULL for Qwen)
        if (cfg.scoring_func == 1) {
            snprintf(name, sizeof(name), "model.layers.%d.%s.e_score_correction_bias", i, cfg.moe_prefix);
            lc->routing_bias = get_tensor_ptr(wf, name);
        }

        // Shared expert weights (NULL when shared_intermediate == 0, e.g. MiniMax)
        if (cfg.shared_intermediate > 0) {
            snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.weight", i);
            lc->sg_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.scales", i);
            lc->sg_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.biases", i);
            lc->sg_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.weight", i);
            lc->su_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.scales", i);
            lc->su_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.biases", i);
            lc->su_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.weight", i);
            lc->sd_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.scales", i);
            lc->sd_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.biases", i);
            lc->sd_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.weight", i);
            lc->seg_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.scales", i);
            lc->seg_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.biases", i);
            lc->seg_b = get_tensor_ptr(wf, name);
        }
    }

    layer_cache_built = 1;
    printf("[cache] Pre-computed weight pointers for %d layers\n", cfg.num_layers);
}

// ============================================================================
// Deferred expert state: holds state for async GPU expert compute.
// GPU experts are submitted async (commit without wait), and the wait+combine
// happens at the start of the NEXT layer. This overlaps ~1ms of GPU expert
// compute with the next layer's attention+routing CPU/GPU work.
// ============================================================================

typedef struct {
    int active;                         // 1 if there's a deferred GPU expert to wait for
    int gpu_combined;                   // 1 if CMD3 includes combine+residual+norm on GPU
                                        // (next layer can skip deferred_wait+finalize+input_norm
                                        //  and submit CMD1 immediately -- buf_input is ready)
    id<MTLCommandBuffer> cmd_experts;   // the async command buffer (committed but not waited)
    float expert_weights[MAX_K];        // routing weights for weighted accumulation
    int valid[MAX_K];                   // which experts loaded successfully
    int actual_K;                       // number of experts
    float *h_mid;                           // [hidden_dim] saved h_mid for final combine
    float shared_gate_score;            // saved shared expert gate score
    float *hidden;                      // pointer to hidden state (for writing final result)
    int layer_idx;                      // which layer produced this deferred state
} DeferredExpertState;

static DeferredExpertState g_deferred = { .active = 0, .h_mid = NULL };

// Wait for the deferred GPU expert command buffer to complete.
// Split from finalize so timing can be measured independently.
static void wait_deferred_experts_gpu(void) {
    if (!g_deferred.active) return;
    [g_deferred.cmd_experts waitUntilCompleted];
}

// CPU readback + accumulate + combine after GPU is done.
// Must be called after wait_deferred_experts_gpu().
// When gpu_combined=1, the GPU already computed the combine+residual+norm
// in CMD3, so we just need to read back the hidden state from buf_moe_hidden.
static void finalize_deferred_experts(void) {
    if (!g_deferred.active) return;

    if (g_deferred.gpu_combined) {
        // GPU-side combine: hidden state is already in buf_moe_hidden.
        // buf_input already has the normalized input for the next layer's CMD1.
        // Just read back hidden (needed for the residual connection in future layers).
        memcpy(g_deferred.hidden, [g_metal->buf_moe_hidden contents],
               cfg.hidden_dim * sizeof(float));
    } else {
        // CPU-side combine (original path)
        // Read back and accumulate routed expert outputs
        float moe_out[cfg.hidden_dim];
        memset(moe_out, 0, sizeof(moe_out));
        for (int k = 0; k < g_deferred.actual_K; k++) {
            if (!g_deferred.valid[k]) continue;
            float *expert_result = (float *)[g_metal->buf_multi_expert_out[k] contents];
            cpu_vec_madd(moe_out, expert_result, g_deferred.expert_weights[k], cfg.hidden_dim);
        }

        if (cfg.shared_intermediate > 0) {
            // Read shared expert result
            float shared_out[cfg.hidden_dim];
            memcpy(shared_out, [g_metal->buf_shared_out contents], cfg.hidden_dim * sizeof(float));

            // Apply shared expert gate
            float shared_weight = cpu_sigmoid(g_deferred.shared_gate_score);
            for (int i = 0; i < cfg.hidden_dim; i++) shared_out[i] *= shared_weight;

            // Final combine: hidden = h_mid + moe_out + shared_out
            for (int i = 0; i < cfg.hidden_dim; i++)
                g_deferred.hidden[i] = g_deferred.h_mid[i] + moe_out[i] + shared_out[i];
        } else {
            // No shared expert: hidden = h_mid + moe_out
            for (int i = 0; i < cfg.hidden_dim; i++)
                g_deferred.hidden[i] = g_deferred.h_mid[i] + moe_out[i];
        }
    }

    g_deferred.active = 0;
    g_deferred.gpu_combined = 0;
    g_deferred.cmd_experts = nil;
}

// Complete the deferred GPU expert compute: wait for GPU, read back, accumulate, combine.
// Must be called before the next layer modifies static scratch buffers.
void complete_deferred_experts(void) {
    wait_deferred_experts_gpu();
    finalize_deferred_experts();
}

// Discard the deferred GPU expert result: wait for GPU to finish (for buffer safety)
// but skip the CPU readback/combine. Used during prefill for intermediate tokens
// where the hidden state will be immediately overwritten by the next token's embedding.
// This saves ~0.1-0.2ms per prefill token (avoids unnecessary memcpy + combine work).
void discard_deferred_experts(void) {
    wait_deferred_experts_gpu();
    // Clear deferred state without reading back results
    if (g_deferred.active) {
        g_deferred.active = 0;
        g_deferred.gpu_combined = 0;
        g_deferred.cmd_experts = nil;
    }
}

// ============================================================================
// Fused layer forward: GPU/CPU overlap + deferred expert pipeline
//
// Pipeline per layer (3 cmd buffers, GPU-side combine in CMD3):
//
//   FAST PATH (when previous CMD3 did GPU-side combine):
//     CMD1: submit immediately (buf_input already populated by CMD3(N-1))
//     WAIT: CMD1 complete (implies CMD3(N-1) also done, queue is serial)
//     CPU:  finalize deferred (read back hidden from buf_moe_hidden)
//
//   SLOW PATH (first layer, or last layer's CMD3 without GPU combine):
//     [DEFERRED] Wait for PREVIOUS layer's CMD3 (if any) + CPU combine
//     CPU:  input_norm(hidden) -> normed -> buf_input
//     CMD1: attention projections (commit)
//     WAIT: CMD1 complete
//
//   Then (both paths):
//     CPU:  attention compute (RoPE/softmax/delta-net)
//     CMD2: o_proj + residual + norm + routing + shared expert projs (8 encoders, 1 commit)
//     WAIT: CMD2 complete
//     CPU:  softmax + top-K routing
//     I/O:  parallel pread K experts (4 pthreads)
//     CMD3: K expert forwards + shared SwiGLU + shared down
//           + moe_combine_residual + rms_norm -> buf_input (ASYNC commit, NO wait)
//     RETURN: GPU experts + combine running async
//
// GPU-side combine eliminates the 0.83ms deferred_wait + CPU combine + input_norm
// at the start of each layer, allowing CMD1 to be submitted immediately.
//
// Key optimizations:
//   1. Parallel pread (4 threads) instead of sequential: ~4x I/O speedup
//   2. o_proj fused into CMD2 with routing (saves 1 commit+wait)
//   3. Deferred CMD3 (expert GPU compute overlapped with next layer)
//   4. GPU-side combine in CMD3 (eliminates CPU deferred_wait + combine + norm)
// ============================================================================

// Static scratch buffers — allocated once, reused across all 40 layers per token.
// Eliminates ~20 malloc/free per layer = ~1200 alloc/free per token.
static float *s_normed    = NULL;   // [cfg.hidden_dim]
static float *s_residual  = NULL;   // [cfg.hidden_dim]
static float *s_attn_proj = NULL;   // [cfg.hidden_dim]
static float *s_h_post    = NULL;   // [cfg.hidden_dim]
static float *s_h_mid     = NULL;   // [cfg.hidden_dim]
static float *s_gate_scores = NULL; // [cfg.num_experts]
static float *s_spec_gate_scores = NULL; // [cfg.num_experts] speculative routing scratch
static int s_spec_indices[MAX_K];         // speculative routing predicted expert indices
static int s_spec_count = 0;              // number of speculative predictions this layer
static float *s_shared_gate = NULL; // [cfg.shared_intermediate]
static float *s_shared_up  = NULL;  // [cfg.shared_intermediate]
static float *s_moe_out   = NULL;   // [cfg.hidden_dim]
static float *s_shared_out = NULL;  // [cfg.hidden_dim]
// Full attention scratch
static float *s_q_proj_out = NULL;  // [cfg.num_attn_heads * cfg.head_dim * 2]
static float *s_k_proj_out = NULL;  // [cfg.num_kv_heads * cfg.head_dim]
static float *s_v_proj_out = NULL;  // [cfg.num_kv_heads * cfg.head_dim]
static float *s_q         = NULL;   // [cfg.num_attn_heads * cfg.head_dim]
static float *s_q_gate    = NULL;   // [cfg.num_attn_heads * cfg.head_dim]
static float *s_attn_out  = NULL;   // [cfg.num_attn_heads * cfg.head_dim]
// Linear attention scratch
static float *s_qkv_proj_out = NULL;   // [cfg.linear_conv_dim]
static float *s_z_proj_out   = NULL;   // [cfg.linear_total_value]
static float *s_beta_proj_out = NULL;  // [cfg.linear_num_v_heads]
static float *s_alpha_proj_out = NULL; // [cfg.linear_num_v_heads]
static float *s_conv_out  = NULL;   // [cfg.linear_conv_dim]
static float *s_out_vals  = NULL;   // [cfg.linear_total_value]
static float *s_gated_out = NULL;   // [cfg.linear_total_value]

static void init_layer_scratch(void) {
    if (s_normed) return;  // already initialized
    s_normed     = calloc(cfg.hidden_dim, sizeof(float));
    s_residual   = calloc(cfg.hidden_dim, sizeof(float));
    s_attn_proj  = calloc(cfg.hidden_dim, sizeof(float));
    s_h_post     = calloc(cfg.hidden_dim, sizeof(float));
    s_h_mid      = calloc(cfg.hidden_dim, sizeof(float));
    s_gate_scores = calloc(cfg.num_experts, sizeof(float));
    s_spec_gate_scores = calloc(cfg.num_experts, sizeof(float));
    s_shared_gate = cfg.shared_intermediate > 0 ? calloc(cfg.shared_intermediate, sizeof(float)) : NULL;
    s_shared_up   = cfg.shared_intermediate > 0 ? calloc(cfg.shared_intermediate, sizeof(float)) : NULL;
    s_moe_out    = calloc(cfg.hidden_dim, sizeof(float));
    s_shared_out = calloc(cfg.hidden_dim, sizeof(float));
    int q_proj_sz = cfg.has_attn_gate ? cfg.num_attn_heads * cfg.head_dim * 2
                                      : cfg.num_attn_heads * cfg.head_dim;
    s_q_proj_out = calloc(q_proj_sz, sizeof(float));
    s_k_proj_out = calloc(cfg.num_kv_heads * cfg.head_dim, sizeof(float));
    s_v_proj_out = calloc(cfg.num_kv_heads * cfg.head_dim, sizeof(float));
    s_q          = calloc(cfg.num_attn_heads * cfg.head_dim, sizeof(float));
    s_q_gate     = cfg.has_attn_gate ? calloc(cfg.num_attn_heads * cfg.head_dim, sizeof(float)) : NULL;
    s_attn_out   = calloc(cfg.num_attn_heads * cfg.head_dim, sizeof(float));
    s_qkv_proj_out = calloc(cfg.linear_conv_dim, sizeof(float));
    s_z_proj_out   = calloc(cfg.linear_total_value, sizeof(float));
    s_beta_proj_out = calloc(cfg.linear_num_v_heads, sizeof(float));
    s_alpha_proj_out = calloc(cfg.linear_num_v_heads, sizeof(float));
    s_conv_out   = calloc(cfg.linear_conv_dim, sizeof(float));
    s_out_vals   = calloc(cfg.linear_total_value, sizeof(float));
    s_gated_out  = calloc(cfg.linear_total_value, sizeof(float));
}

// Pre-computed projection results for batched prefill (NULL = compute projections normally)
typedef struct {
    float *proj[4];  // Full attn: [0]=Q, [1]=K, [2]=V; Linear: [0]=QKV, [1]=Z, [2]=beta, [3]=alpha
    int dims[4];     // Output dimension for each projection
    int count;       // Number of projections (3 for full, 4 for linear)
} PrecomputedProj;

static void fused_layer_forward_ex(
    WeightFile *wf,
    int layer_idx,
    float *hidden,
    KVCache *kv,
    LinearAttnState *la_state,
    int pos,
    const void *mmap_base,
    int K,
    int packed_fd,
    PrecomputedProj *precomp  // NULL = compute projections, non-NULL = skip CMD1
);

// Original interface (all existing call sites use this)
void fused_layer_forward(
    WeightFile *wf,
    int layer_idx,
    float *hidden,           // [cfg.hidden_dim] in/out
    KVCache *kv,             // non-NULL for full attention layers
    LinearAttnState *la_state, // non-NULL for linear attention layers
    int pos,                 // position for RoPE
    const void *mmap_base,   // mmap'd layer file (NULL if not available)
    int K,                   // number of active experts
    int packed_fd            // fd for packed expert file
) {
    fused_layer_forward_ex(wf, layer_idx, hidden, kv, la_state, pos, mmap_base, K, packed_fd, NULL);
}

static void fused_layer_forward_ex(
    WeightFile *wf,
    int layer_idx,
    float *hidden,
    KVCache *kv,
    LinearAttnState *la_state,
    int pos,
    const void *mmap_base,
    int K,
    int packed_fd,
    PrecomputedProj *precomp
) {
    // Per-layer quant: temporarily switch to the per-layer expert format
    int saved_use_2bit = g_use_2bit;
    int saved_use_q3_experts = g_use_q3_experts;
    int saved_use_q3_outlier = g_use_q3_outlier;
    ExpertLayout saved_active_q3_layout = g_active_q3_layout;
    int saved_active_q3_layout_valid = g_active_q3_layout_valid;
    if (saved_use_2bit) g_use_2bit = g_layer_is_2bit[layer_idx];
    if (saved_use_q3_experts) {
        g_use_q3_outlier = g_layer_is_q3_outlier[layer_idx];
        g_use_q3_experts = g_layer_is_q3_hybrid[layer_idx] || g_layer_is_q3_outlier[layer_idx];
        if ((g_use_q3_experts || g_use_q3_outlier) &&
            g_q3_layout_manifest_loaded &&
            g_q3_layer_layout_valid[layer_idx]) {
            g_active_q3_layout = g_q3_layer_layouts[layer_idx];
            g_active_q3_layout_valid = 1;
        } else {
            g_active_q3_layout_valid = 0;
        }
    }

    double t_layer_start = 0, t0 = 0, t1 = 0;
    if (g_timing_enabled) { t_layer_start = now_ms(); }
    int pred_started = 0;  // set to 1 if we started prediction preads during CMD1_wait

    init_layer_scratch();
    if (!layer_cache_built) build_layer_cache(wf);
    LayerWeightCache *lc = &layer_cache[layer_idx];
    int is_full = (kv != NULL);

    // =====================================================================
    // PHASE 1: Deferred completion + CMD1 (attention projections)
    // =====================================================================

    // ---- Prepare attention projection specs (doesn't depend on hidden) ----
    int num_attn_specs = 0;
    BatchMatvecSpec attn_specs[5];
    float *q_proj_out = NULL, *k_out = NULL, *v_out = NULL;
    float *qkv_out = NULL, *z_out = NULL, *beta_out = NULL, *alpha_out = NULL;

    if (is_full) {
        int q_dim_full = cfg.num_attn_heads * cfg.head_dim;
        int q_proj_dim = cfg.has_attn_gate ? q_dim_full * 2 : q_dim_full;
        int kv_dim = cfg.num_kv_heads * cfg.head_dim;

        q_proj_out = s_q_proj_out;
        k_out = s_k_proj_out;
        v_out = s_v_proj_out;

        if (lc->q_w && lc->q_s && lc->q_b && lc->k_w && lc->k_s && lc->k_b &&
            lc->v_w && lc->v_s && lc->v_b) {
            attn_specs[0] = (BatchMatvecSpec){ lc->q_w, lc->q_s, lc->q_b, q_proj_out, (uint32_t)q_proj_dim, cfg.hidden_dim, cfg.group_size, 0 };
            attn_specs[1] = (BatchMatvecSpec){ lc->k_w, lc->k_s, lc->k_b, k_out,      (uint32_t)kv_dim,     cfg.hidden_dim, cfg.group_size, 1 };
            attn_specs[2] = (BatchMatvecSpec){ lc->v_w, lc->v_s, lc->v_b, v_out,      (uint32_t)kv_dim,     cfg.hidden_dim, cfg.group_size, 2 };
            num_attn_specs = 3;
        }
    } else {
        int qkv_dim = cfg.linear_conv_dim;
        int z_dim = cfg.linear_total_value;

        qkv_out = s_qkv_proj_out;
        z_out = s_z_proj_out;
        beta_out = s_beta_proj_out;
        alpha_out = s_alpha_proj_out;

        if (lc->qkv_w && lc->qkv_s && lc->qkv_b && lc->z_w && lc->z_s && lc->z_b &&
            lc->b_w && lc->b_s && lc->b_b && lc->a_w && lc->a_s && lc->a_b) {
            attn_specs[0] = (BatchMatvecSpec){ lc->qkv_w, lc->qkv_s, lc->qkv_b, qkv_out,   (uint32_t)qkv_dim,            cfg.hidden_dim, cfg.group_size, 0 };
            attn_specs[1] = (BatchMatvecSpec){ lc->z_w,   lc->z_s,   lc->z_b,   z_out,      (uint32_t)z_dim,              cfg.hidden_dim, cfg.group_size, 1 };
            attn_specs[2] = (BatchMatvecSpec){ lc->b_w,   lc->b_s,   lc->b_b,   beta_out,   (uint32_t)cfg.linear_num_v_heads, cfg.hidden_dim, cfg.group_size, 2 };
            attn_specs[3] = (BatchMatvecSpec){ lc->a_w,   lc->a_s,   lc->a_b,   alpha_out,  (uint32_t)cfg.linear_num_v_heads, cfg.hidden_dim, cfg.group_size, 3 };
            num_attn_specs = 4;
        }
    }

    // ---- Deferred completion + CMD1 (sequential) ----
    float *normed = s_normed;
    float *residual = s_residual;
    id<MTLCommandBuffer> cmd1 = nil;
    int gpu_linear_attn = 0;  // set to 1 if GPU handles entire linear attention pipeline
    int cmd1_cmd2_merged = 0; // set to 1 if CMD2 work is merged into CMD1 (linear attn optimization)

    // Pre-compute linear_layer_idx (needed in Phase 2 for delta-net)
    int linear_layer_idx = -1;
    if (!is_full) {
        linear_layer_idx = cfg.linear_index[layer_idx];
    }

    // Variables declared here to allow goto to skip over CMD1
    dispatch_group_t spec_group = NULL;
    int spec_preload_count = 0;
    int spec_routing_enabled = 0;  // DISABLED: cache pollution + overhead makes it slower

    // ---- PRECOMPUTED PROJECTION SKIP ----
    // If precomp is set, projections were already computed by batched GEMM.
    // Copy results into scratch buffers, handle deferred completion, skip CMD1.
    if (precomp) {
        // Complete deferred experts from previous layer
        if (g_timing_enabled) { t0 = now_ms(); }
        wait_deferred_experts_gpu();
        if (g_timing_enabled) { t1 = now_ms(); g_timing.deferred_wait += t1 - t0; }
        if (g_timing_enabled) { t0 = now_ms(); }
        finalize_deferred_experts();
        if (g_timing_enabled) { t1 = now_ms(); g_timing.deferred_cpu += t1 - t0; }

        // Residual = hidden before attention
        cpu_vec_copy(residual, hidden, HIDDEN_DIM);

        // Compute input norm and upload to buf_input (needed by CMD2 fused path)
        cpu_rms_norm(hidden, lc->input_norm_w, normed, HIDDEN_DIM, RMS_NORM_EPS);
        if (g_metal && g_metal->buf_input) {
            memcpy([g_metal->buf_input contents], normed, HIDDEN_DIM * sizeof(float));
        }

        // Copy pre-computed projections into scratch buffers
        if (is_full) {
            memcpy(q_proj_out, precomp->proj[0], precomp->dims[0] * sizeof(float));
            memcpy(k_out,      precomp->proj[1], precomp->dims[1] * sizeof(float));
            memcpy(v_out,      precomp->proj[2], precomp->dims[2] * sizeof(float));
        } else {
            memcpy(qkv_out,    precomp->proj[0], precomp->dims[0] * sizeof(float));
            memcpy(z_out,      precomp->proj[1], precomp->dims[1] * sizeof(float));
            memcpy(beta_out,   precomp->proj[2], precomp->dims[2] * sizeof(float));
            memcpy(alpha_out,  precomp->proj[3], precomp->dims[3] * sizeof(float));
        }

        // Skip CMD1 entirely — jump to PHASE 2 (attention compute)
        goto phase2_attention;
    }

    // Can we run the full linear attention pipeline on GPU in CMD1?
    int can_gpu_linear = (gpu_linear_attn_enabled &&
                          !is_full && g_metal && g_metal->delta_net_step &&
                          g_metal->conv1d_step && g_metal->rms_norm_qk &&
                          g_metal->compute_decay_beta && g_metal->gated_rms_norm &&
                          g_metal->wf_buf &&
                          linear_layer_idx >= 0 && linear_layer_idx < cfg.num_linear_layers &&
                          lc->conv1d_w && lc->A_log && lc->dt_bias && lc->gated_norm_w &&
                          !linear_attn_bypass);

    // Check if previous layer's CMD3 already computed combine+residual+norm on GPU.
    // If so, buf_input already contains the normalized input for this layer's CMD1.
    // We can submit CMD1 immediately — the GPU queue serializes CMD3(N-1) then CMD1(N).
    int prev_gpu_combined = (g_deferred.active && g_deferred.gpu_combined);

    if (prev_gpu_combined && g_metal && g_metal->wf_buf && num_attn_specs > 0) {
        // ---- FAST PATH: GPU-combined previous CMD3 ----
        // buf_input already has the normalized hidden state from CMD3(N-1).
        // Submit CMD1 immediately — GPU runs CMD3(N-1) then CMD1(N) back-to-back.
        if (g_timing_enabled) { t0 = now_ms(); }

        cmd1 = [g_metal->queue commandBuffer];
        gpu_encode_batch_matvec(g_metal, cmd1, attn_specs, num_attn_specs);

        // GPU linear attention: encode conv1d + normalize + decay/beta + delta-net + gated_norm into CMD1
        if (can_gpu_linear && num_attn_specs == 4) {
            // batch_out[0]=qkv(12288), [1]=z(8192), [2]=beta(64), [3]=alpha(64)
            uint32_t conv_dim = cfg.linear_conv_dim;
            // Enc L1: conv1d_step — input=batch_out[0], weights=conv1d_w, state=buf_conv_state, output=buf_conv_output
            // NOTE: staging was already reset by gpu_encode_batch_matvec above.
            // All tensors below accumulate into the same staging buffer — no resets
            // until cmd1 is committed, to avoid overwriting data the GPU hasn't read yet.
            {
                id<MTLBuffer> conv1d_w_buf; NSUInteger conv1d_w_off;
                metal_find_chunk_sized(g_metal, lc->conv1d_w,
                    (size_t)cfg.linear_conv_dim * cfg.linear_conv_dim * 2,
                    &conv1d_w_buf, &conv1d_w_off);
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:g_metal->conv1d_step];
                [enc setBuffer:g_metal->buf_conv_state[linear_layer_idx] offset:0 atIndex:0];
                [enc setBuffer:g_metal->batch_out[0]    offset:0            atIndex:1]; // qkv projection output
                [enc setBuffer:conv1d_w_buf offset:conv1d_w_off atIndex:2]; // conv weights (bf16)
                [enc setBuffer:g_metal->buf_conv_output offset:0            atIndex:3]; // conv output
                [enc setBytes:&conv_dim length:4 atIndex:4];
                uint32_t tgs = (conv_dim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }

            // Enc L2: rms_norm_qk — normalize q and k in conv_output in-place
            {
                uint32_t key_dim = cfg.linear_key_dim;  // 128
                float inv_scale = 1.0f / sqrtf((float)cfg.linear_key_dim);
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:g_metal->rms_norm_qk];
                [enc setBuffer:g_metal->buf_conv_output offset:0 atIndex:0];  // q at offset 0
                [enc setBuffer:g_metal->buf_conv_output offset:cfg.linear_total_key * sizeof(float) atIndex:1];  // k at offset 2048 floats
                [enc setBytes:&key_dim   length:4 atIndex:2];
                [enc setBytes:&inv_scale length:4 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(cfg.linear_num_k_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(cfg.linear_key_dim, 1, 1)];
                [enc endEncoding];
            }

            // Enc L3: compute_decay_beta — alpha=batch_out[3], beta=batch_out[2], A_log+dt_bias from wf_buf
            {
                id<MTLBuffer> alog_buf, dtb_buf; NSUInteger alog_off, dtb_off;
                metal_find_chunk_sized(g_metal, lc->A_log,
                    (size_t)cfg.linear_num_v_heads * 4, &alog_buf, &alog_off);  // float
                metal_find_chunk_sized(g_metal, lc->dt_bias,
                    (size_t)cfg.linear_num_v_heads * 2, &dtb_buf, &dtb_off);    // bf16
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:g_metal->compute_decay_beta];
                [enc setBuffer:g_metal->batch_out[3]       offset:0          atIndex:0]; // alpha
                [enc setBuffer:g_metal->batch_out[2]       offset:0          atIndex:1]; // beta
                [enc setBuffer:alog_buf  offset:alog_off  atIndex:2]; // A_log
                [enc setBuffer:dtb_buf   offset:dtb_off   atIndex:3]; // dt_bias (bf16)
                [enc setBuffer:g_metal->buf_delta_g_decay  offset:0          atIndex:4]; // g_decay output
                [enc setBuffer:g_metal->buf_delta_beta     offset:0          atIndex:5]; // beta_gate output
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(cfg.linear_num_v_heads, 1, 1)];
                [enc endEncoding];
            }

            // Enc L4: gated_delta_net_step — the main recurrence
            {
                uint32_t khpv = cfg.linear_num_v_heads / cfg.linear_num_k_heads;  // 4
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:g_metal->delta_net_step_fused ? g_metal->delta_net_step_fused : g_metal->delta_net_step];
                [enc setBuffer:g_metal->buf_delta_state[linear_layer_idx] offset:0 atIndex:0]; // persistent state
                [enc setBuffer:g_metal->buf_conv_output offset:0 atIndex:1]; // q (first 2048 floats)
                [enc setBuffer:g_metal->buf_conv_output offset:cfg.linear_total_key * sizeof(float) atIndex:2]; // k (next 2048)
                [enc setBuffer:g_metal->buf_conv_output offset:2 * cfg.linear_total_key * sizeof(float) atIndex:3]; // v (next 8192)
                [enc setBuffer:g_metal->buf_delta_g_decay offset:0 atIndex:4];
                [enc setBuffer:g_metal->buf_delta_beta    offset:0 atIndex:5];
                [enc setBuffer:g_metal->buf_delta_output  offset:0 atIndex:6]; // output [8192]
                [enc setBytes:&khpv length:sizeof(khpv) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(cfg.linear_num_v_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding];
            }

            // Enc L5: gated_rms_norm — normalize+gate delta-net output -> batch_out[6] for CMD2 o_proj
            {
                id<MTLBuffer> gnw_buf; NSUInteger gnw_off;
                metal_find_chunk_sized(g_metal, lc->gated_norm_w,
                    (size_t)cfg.linear_total_value * 2, &gnw_buf, &gnw_off);  // bf16
                uint32_t value_dim = cfg.linear_value_dim;  // 128
                float eps = cfg.rms_norm_eps;
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:g_metal->gated_rms_norm];
                [enc setBuffer:g_metal->buf_delta_output offset:0          atIndex:0]; // values [8192]
                [enc setBuffer:g_metal->batch_out[1]     offset:0          atIndex:1]; // z (z projection output) [8192]
                [enc setBuffer:gnw_buf offset:gnw_off atIndex:2]; // weight (bf16)
                [enc setBuffer:g_metal->batch_out[6]     offset:0          atIndex:3]; // output -> batch_out[6] for CMD2
                [enc setBytes:&value_dim length:4 atIndex:4];
                [enc setBytes:&eps       length:4 atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(cfg.linear_num_v_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(cfg.linear_value_dim, 1, 1)];
                [enc endEncoding];
            }

            gpu_linear_attn = 1;
        }

        // ---- CMD1+CMD2 merge for linear attention layers ----
        // When gpu_linear_attn is active, CMD2 (o_proj + residual + norm + routing)
        // can be encoded directly into CMD1, eliminating one commit+wait cycle.
        // buf_moe_hidden (from CMD3(N-1)) is used as the residual source on GPU —
        // serial queue ordering guarantees CMD3(N-1) completes before CMD1 starts.
        if (g_cmd_merge_enabled && gpu_linear_attn && g_metal->wf_buf &&
            lc->gate_w && lc->gate_s && lc->gate_b &&
            lc->sg_w && lc->sg_s && lc->sg_b &&
            lc->su_w && lc->su_s && lc->su_b &&
            lc->seg_w && lc->seg_s && lc->seg_b &&
            g_metal->residual_add && g_metal->rms_norm_sum &&
            g_metal->rms_norm_apply_bf16 && lc->post_attn_norm_w &&
            g_deferred.gpu_combined && g_metal->buf_moe_hidden) {
            // batch_out[6] has gated_rms_norm result from CMD1.
            // buf_moe_hidden has pre-attention hidden (residual) from CMD3(N-1).

            // o_proj matvec into CMD1
            {
                uint32_t o_out_dim = cfg.hidden_dim;
                uint32_t o_in_dim = (uint32_t)cfg.linear_total_value;
                uint32_t o_gs = cfg.group_size;
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:g_metal->matvec_fast];
                [enc setBuffer:g_metal->wf_buf offset:(NSUInteger)((const char *)lc->out_proj_w - (const char *)[g_metal->wf_buf contents]) atIndex:0];
                [enc setBuffer:g_metal->wf_buf offset:(NSUInteger)((const char *)lc->out_proj_s - (const char *)[g_metal->wf_buf contents]) atIndex:1];
                [enc setBuffer:g_metal->wf_buf offset:(NSUInteger)((const char *)lc->out_proj_b - (const char *)[g_metal->wf_buf contents]) atIndex:2];
                [enc setBuffer:g_metal->batch_out[6] offset:0 atIndex:3];
                [enc setBuffer:g_metal->buf_output   offset:0 atIndex:4];
                [enc setBytes:&o_out_dim  length:4 atIndex:5];
                [enc setBytes:&o_in_dim   length:4 atIndex:6];
                [enc setBytes:&o_gs       length:4 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(o_out_dim, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                [enc endEncoding];
            }
            // residual_add (buf_output + buf_moe_hidden -> buf_h_mid)
            {
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                uint32_t dim = cfg.hidden_dim;
                [enc setComputePipelineState:g_metal->residual_add];
                [enc setBuffer:g_metal->buf_moe_hidden offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_output     offset:0 atIndex:1];
                [enc setBuffer:g_metal->buf_h_mid      offset:0 atIndex:2];
                [enc setBytes:&dim length:4 atIndex:3];
                uint32_t tgs = (dim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
            // rms_norm_sum_sq (buf_h_mid -> buf_sum_sq)
            {
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                uint32_t dim = cfg.hidden_dim;
                [enc setComputePipelineState:g_metal->rms_norm_sum];
                [enc setBuffer:g_metal->buf_h_mid  offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_sum_sq offset:0 atIndex:1];
                [enc setBytes:&dim length:4 atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
            // rms_norm_apply_bf16 (buf_h_mid + norm_w -> buf_input)
            {
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                uint32_t dim = cfg.hidden_dim;
                float eps = cfg.rms_norm_eps;
                [enc setComputePipelineState:g_metal->rms_norm_apply_bf16];
                [enc setBuffer:g_metal->buf_h_mid  offset:0       atIndex:0];
                [enc setBuffer:g_metal->wf_buf offset:(NSUInteger)((const char *)lc->post_attn_norm_w - (const char *)[g_metal->wf_buf contents]) atIndex:1];
                [enc setBuffer:g_metal->buf_sum_sq offset:0       atIndex:2];
                [enc setBuffer:g_metal->buf_input  offset:0       atIndex:3];
                [enc setBytes:&dim length:4 atIndex:4];
                [enc setBytes:&eps length:4 atIndex:5];
                uint32_t tgs = (dim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
            // routing + shared expert projections
            // Slots 0-3: safe to reuse because attention projection results in
            // batch_out[0-3] have already been consumed by the GPU linear attention
            // pipeline (conv1d, delta-net, etc.) earlier in this same CMD1.
            // Metal command encoders execute sequentially within a command buffer.
            {
                BatchMatvecSpec moe_specs_merged[4] = {
                    { lc->gate_w, lc->gate_s, lc->gate_b, s_gate_scores,        (uint32_t)cfg.num_experts,         cfg.hidden_dim, cfg.group_size, 0 },
                    { lc->sg_w,   lc->sg_s,   lc->sg_b,   s_shared_gate,        (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 1 },
                    { lc->su_w,   lc->su_s,   lc->su_b,   s_shared_up,          (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 2 },
                    { lc->seg_w,  lc->seg_s,  lc->seg_b,  &g_merged_shared_gate_score, 1,                         cfg.hidden_dim, cfg.group_size, 3 },
                };
                gpu_encode_batch_matvec(g_metal, cmd1, moe_specs_merged, 4);
            }

            cmd1_cmd2_merged = 1;
        }

        [cmd1 commit];

        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd1_submit += t1 - t0; }

        // Wait for CMD1 (implies CMD3(N-1) also done, since queue is serial)
        if (g_timing_enabled) { t0 = now_ms(); }
        [cmd1 waitUntilCompleted];
        if (!gpu_linear_attn) {
            gpu_flush_batch_results(g_metal, attn_specs, num_attn_specs);
        }
        if (cmd1_cmd2_merged) {
            // Read back merged CMD2 results (slots 0-3, matching encode above)
            float m_sgs = 0.0f;
            BatchMatvecSpec moe_rb[4] = {
                { NULL, NULL, NULL, s_gate_scores,   (uint32_t)cfg.num_experts,         cfg.hidden_dim, cfg.group_size, 0 },
                { NULL, NULL, NULL, s_shared_gate,   (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 1 },
                { NULL, NULL, NULL, s_shared_up,     (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 2 },
                { NULL, NULL, NULL, &m_sgs,            1,                              cfg.hidden_dim, cfg.group_size, 3 },
            };
            gpu_flush_batch_results(g_metal, moe_rb, 4);
            g_merged_shared_gate_score = m_sgs;
            // Read h_mid and h_post from GPU
            memcpy(s_h_mid, [g_metal->buf_h_mid contents], cfg.hidden_dim * sizeof(float));
            memcpy(s_h_post, [g_metal->buf_input contents], cfg.hidden_dim * sizeof(float));
            memcpy(hidden, s_h_mid, cfg.hidden_dim * sizeof(float));
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd1_wait += t1 - t0; }

        // Now CMD3(N-1) is done. Read back hidden state from GPU.
        if (g_timing_enabled) { t0 = now_ms(); }
        if (!cmd1_cmd2_merged) {
            finalize_deferred_experts();  // reads buf_moe_hidden -> hidden
        } else {
            // Merged path: hidden already set from buf_h_mid above.
            g_deferred.active = 0;
            g_deferred.cmd_experts = nil;
        }

        // Start predicted expert preads AFTER CMD1_wait.
        // CMD3(N-1) is guaranteed done (serial queue), so buf_B is safe to overwrite.
        // Predictions overlap with CPU attn + CMD2 + routing (~0.6ms head start).
        // Predicted experts that hit page cache (same as previous token) complete in ~0.1ms.
        if (g_pred_enabled && g_pred_generating && g_pred_valid && packed_fd >= 0 &&
            g_metal->buf_multi_expert_data_B[0] && PRED_COUNT(layer_idx) > 0) {
            async_pread_start(packed_fd, &PRED_EXPERT(layer_idx, 0),
                              PRED_COUNT(layer_idx),
                              g_metal->buf_multi_expert_data_B, mmap_base,
                              layer_idx);
            pred_started = 1;
        }
        // Set up residual for CMD2 (residual = hidden before this layer's attention)
        cpu_vec_copy(residual, hidden, cfg.hidden_dim);
        if (g_timing_enabled) { t1 = now_ms(); g_timing.deferred_cpu += t1 - t0; }

        // No input_norm needed — CMD3 already computed it into buf_input.
        // normed is only needed if speculative routing is enabled (currently disabled).
        // Skip the readback to avoid unnecessary overhead.
    } else {
        // ---- ORIGINAL PATH: CPU deferred completion + input norm ----
        // Complete deferred experts from previous layer
        if (g_timing_enabled) { t0 = now_ms(); }
        wait_deferred_experts_gpu();
        if (g_timing_enabled) { t1 = now_ms(); g_timing.deferred_wait += t1 - t0; }

        if (g_timing_enabled) { t0 = now_ms(); }
        finalize_deferred_experts();
        if (g_timing_enabled) { t1 = now_ms(); g_timing.deferred_cpu += t1 - t0; }

        // Input norm
        if (g_timing_enabled) { t0 = now_ms(); }
        cpu_vec_copy(residual, hidden, cfg.hidden_dim);
        cpu_rms_norm(hidden, lc->input_norm_w, normed, cfg.hidden_dim, cfg.rms_norm_eps);
        if (g_timing_enabled) { t1 = now_ms(); g_timing.input_norm += t1 - t0; }

        // Submit CMD1: attention projections
        if (g_timing_enabled) { t0 = now_ms(); }
        if (g_metal && g_metal->wf_buf && num_attn_specs > 0) {
            memcpy([g_metal->buf_input contents], normed, cfg.hidden_dim * sizeof(float));
            cmd1 = [g_metal->queue commandBuffer];
            gpu_encode_batch_matvec(g_metal, cmd1, attn_specs, num_attn_specs);

            // GPU linear attention: encode conv1d + normalize + decay/beta + delta-net + gated_norm into CMD1
            if (can_gpu_linear && num_attn_specs == 4) {
                uint32_t conv_dim = cfg.linear_conv_dim;

                // Enc L1: conv1d_step
                // NOTE: staging was reset by gpu_encode_batch_matvec — don't reset again
                {
                    id<MTLBuffer> conv1d_w_buf; NSUInteger conv1d_w_off;
                    metal_find_chunk_sized(g_metal, lc->conv1d_w,
                        (size_t)cfg.linear_conv_dim * cfg.linear_conv_dim * 2,
                        &conv1d_w_buf, &conv1d_w_off);
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->conv1d_step];
                    [enc setBuffer:g_metal->buf_conv_state[linear_layer_idx] offset:0 atIndex:0];
                    [enc setBuffer:g_metal->batch_out[0]    offset:0            atIndex:1];
                    [enc setBuffer:conv1d_w_buf offset:conv1d_w_off atIndex:2];
                    [enc setBuffer:g_metal->buf_conv_output offset:0            atIndex:3];
                    [enc setBytes:&conv_dim length:4 atIndex:4];
                    uint32_t tgs = (conv_dim + 255) / 256;
                    [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                }

                // Enc L2: rms_norm_qk
                {
                    uint32_t key_dim = cfg.linear_key_dim;
                    float inv_scale = 1.0f / sqrtf((float)cfg.linear_key_dim);
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->rms_norm_qk];
                    [enc setBuffer:g_metal->buf_conv_output offset:0 atIndex:0];
                    [enc setBuffer:g_metal->buf_conv_output offset:cfg.linear_total_key * sizeof(float) atIndex:1];
                    [enc setBytes:&key_dim   length:4 atIndex:2];
                    [enc setBytes:&inv_scale length:4 atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(cfg.linear_num_k_heads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(cfg.linear_key_dim, 1, 1)];
                    [enc endEncoding];
                }

                // Enc L3: compute_decay_beta
                {
                    id<MTLBuffer> alog_buf, dtb_buf; NSUInteger alog_off, dtb_off;
                    metal_find_chunk_sized(g_metal, lc->A_log,
                        (size_t)cfg.linear_num_v_heads * 4, &alog_buf, &alog_off);  // float
                    metal_find_chunk_sized(g_metal, lc->dt_bias,
                        (size_t)cfg.linear_num_v_heads * 2, &dtb_buf, &dtb_off);    // bf16
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->compute_decay_beta];
                    [enc setBuffer:g_metal->batch_out[3]       offset:0          atIndex:0];
                    [enc setBuffer:g_metal->batch_out[2]       offset:0          atIndex:1];
                    [enc setBuffer:alog_buf  offset:alog_off  atIndex:2];
                    [enc setBuffer:dtb_buf   offset:dtb_off   atIndex:3];
                    [enc setBuffer:g_metal->buf_delta_g_decay  offset:0          atIndex:4];
                    [enc setBuffer:g_metal->buf_delta_beta     offset:0          atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(cfg.linear_num_v_heads, 1, 1)];
                    [enc endEncoding];
                }

                // Enc L4: gated_delta_net_step
                {
                    uint32_t khpv = cfg.linear_num_v_heads / cfg.linear_num_k_heads;
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->delta_net_step_fused ? g_metal->delta_net_step_fused : g_metal->delta_net_step];
                    [enc setBuffer:g_metal->buf_delta_state[linear_layer_idx] offset:0 atIndex:0];
                    [enc setBuffer:g_metal->buf_conv_output offset:0 atIndex:1];
                    [enc setBuffer:g_metal->buf_conv_output offset:cfg.linear_total_key * sizeof(float) atIndex:2];
                    [enc setBuffer:g_metal->buf_conv_output offset:2 * cfg.linear_total_key * sizeof(float) atIndex:3];
                    [enc setBuffer:g_metal->buf_delta_g_decay offset:0 atIndex:4];
                    [enc setBuffer:g_metal->buf_delta_beta    offset:0 atIndex:5];
                    [enc setBuffer:g_metal->buf_delta_output  offset:0 atIndex:6];
                    [enc setBytes:&khpv length:sizeof(khpv) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake(cfg.linear_num_v_heads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [enc endEncoding];
                }

                // Enc L5: gated_rms_norm -> batch_out[6]
                {
                    id<MTLBuffer> gnw_buf; NSUInteger gnw_off;
                    metal_find_chunk_sized(g_metal, lc->gated_norm_w,
                        (size_t)cfg.linear_total_value * 2, &gnw_buf, &gnw_off);  // bf16
                    uint32_t value_dim = cfg.linear_value_dim;
                    float eps = cfg.rms_norm_eps;
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->gated_rms_norm];
                    [enc setBuffer:g_metal->buf_delta_output offset:0          atIndex:0];
                    [enc setBuffer:g_metal->batch_out[1]     offset:0          atIndex:1];
                    [enc setBuffer:gnw_buf offset:gnw_off atIndex:2];
                    [enc setBuffer:g_metal->batch_out[6]     offset:0          atIndex:3];
                    [enc setBytes:&value_dim length:4 atIndex:4];
                    [enc setBytes:&eps       length:4 atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(cfg.linear_num_v_heads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(cfg.linear_value_dim, 1, 1)];
                    [enc endEncoding];
                }

                gpu_linear_attn = 1;
            }

            [cmd1 commit];
        } else {
            for (int i = 0; i < num_attn_specs; i++) {
                BatchMatvecSpec *s = &attn_specs[i];
                cpu_dequant_matvec(s->W, s->scales, s->biases, normed, s->out_cpu,
                                   s->out_dim, s->in_dim, s->group_size);
            }
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd1_submit += t1 - t0; }

        // Wait for CMD1
        if (g_timing_enabled) { t0 = now_ms(); }
        if (cmd1) {
            [cmd1 waitUntilCompleted];
            if (!gpu_linear_attn) {
                gpu_flush_batch_results(g_metal, attn_specs, num_attn_specs);
            }
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd1_wait += t1 - t0; }
    }

    // =====================================================================
    // SPECULATIVE EARLY ROUTING — overlap expert I/O with CPU attention
    // =====================================================================
    // Compute approximate routing using the PRE-attention normed hidden state.
    // The real routing (in CMD2/PHASE 3) uses the POST-attention state, so this
    // is an approximation. Fire off async pread for predicted cache misses via
    // dispatch_group so the I/O runs concurrently with CPU attention compute.
    // After CPU attention, we wait for the group to finish. When the real routing
    // happens later, predicted experts are already in the LRU cache as hits.

    if (g_timing_enabled) { t0 = now_ms(); }
    s_spec_count = 0;

    if (spec_routing_enabled && (g_expert_cache || g_malloc_cache) && packed_fd >= 0 && lc->gate_w) {
        float *spec_scores = s_spec_gate_scores;
        memset(spec_scores, 0, cfg.num_experts * sizeof(float));

        // Gate projection matvec on pre-attention normed input (CPU, ~0.1ms for 512x4096)
        cpu_dequant_matvec(lc->gate_w, lc->gate_s, lc->gate_b,
                           normed, spec_scores,
                           cfg.num_experts, cfg.hidden_dim, cfg.group_size);

        int spec_K = (K > MAX_K) ? MAX_K : K;
        float spec_weights[MAX_K];
        cpu_route_experts(spec_scores, cfg.num_experts, spec_K,
                          lc->routing_bias, s_spec_indices, spec_weights);
        s_spec_count = spec_K;

        g_spec_route_attempts += spec_K;

        // Initialize GCD queue if needed
        if (!g_io_gcd_queue)
            g_io_gcd_queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);

        // Check cache for each predicted expert, start async I/O for misses
        if (g_malloc_cache) {
            spec_group = dispatch_group_create();
            for (int k = 0; k < spec_K; k++) {
                int eidx = s_spec_indices[k];
                id<MTLBuffer> cached = malloc_cache_lookup(g_malloc_cache, layer_idx, eidx);
                if (!cached) {
                    int cidx = -1;
                    id<MTLBuffer> buf = malloc_cache_insert(g_malloc_cache, layer_idx, eidx, &cidx);
                    if (buf && cidx >= 0) {
                        int fd_copy = packed_fd;
                        void *dst = g_malloc_cache->data[cidx];
                        off_t eoff; size_t esz;
                        expert_offset_size(layer_idx, eidx, &eoff, &esz);
                        off_t offset = eoff;
                        size_t sz = esz;
                        dispatch_group_async(spec_group, g_io_gcd_queue, ^{
                            pread(fd_copy, dst, sz, offset);
                        });
                        spec_preload_count++;
                        g_spec_route_preloads++;
                    }
                }
            }
        } else if (g_expert_cache) {
            spec_group = dispatch_group_create();
            for (int k = 0; k < spec_K; k++) {
                int eidx = s_spec_indices[k];
                id<MTLBuffer> cached = expert_cache_lookup(g_expert_cache, layer_idx, eidx);
                if (!cached) {
                    id<MTLBuffer> buf = expert_cache_insert(g_expert_cache, layer_idx, eidx);
                    if (buf) {
                        int fd_copy = packed_fd;
                        void *dst = [buf contents];
                        off_t eoff; size_t esz;
                        expert_offset_size(layer_idx, eidx, &eoff, &esz);
                        off_t offset = eoff;
                        size_t sz = esz;
                        dispatch_group_async(spec_group, g_io_gcd_queue, ^{
                            pread(fd_copy, dst, sz, offset);
                        });
                        spec_preload_count++;
                        g_spec_route_preloads++;
                    }
                }
            }
        }
    }
    (void)spec_preload_count;  // tracked via g_spec_route_preloads

    if (g_timing_enabled) { t1 = now_ms(); g_timing.spec_route += t1 - t0; }

    // =====================================================================
    // PHASE 2: CPU attention compute
    // =====================================================================
phase2_attention:

    if (g_timing_enabled) { t0 = now_ms(); }

    float *attn_projected = s_attn_proj;
    memset(attn_projected, 0, cfg.hidden_dim * sizeof(float));

    // Pre-lookup o_proj / out_proj weights (used after attention compute)
    // These are looked up NOW to avoid repeated snprintf later.
    uint32_t *oproj_w = NULL;
    uint16_t *oproj_s = NULL, *oproj_b = NULL;
    int oproj_in_dim = 0;

    if (is_full) {
        oproj_w = lc->o_w; oproj_s = lc->o_s; oproj_b = lc->o_b;
        oproj_in_dim = cfg.num_attn_heads * cfg.head_dim;
    } else if (!linear_attn_bypass) {
        oproj_w = lc->out_proj_w; oproj_s = lc->out_proj_s; oproj_b = lc->out_proj_b;
        oproj_in_dim = cfg.linear_total_value;
    }

    // All MoE weight pointers from cache (zero snprintf overhead)
    uint32_t *gate_w = lc->gate_w; uint16_t *gate_s = lc->gate_s, *gate_b = lc->gate_b;
    uint32_t *sgw = lc->sg_w;     uint16_t *sgs = lc->sg_s,       *sgb = lc->sg_b;
    uint32_t *suw = lc->su_w;     uint16_t *sus = lc->su_s,       *sub = lc->su_b;
    uint32_t *seg_w = lc->seg_w;  uint16_t *seg_s = lc->seg_s,   *seg_b = lc->seg_b;
    uint32_t *sdw = lc->sd_w;     uint16_t *sds = lc->sd_s,       *sdb = lc->sd_b;

    // ---- CPU attention compute (produces attn_out for o_proj) ----
    float *attn_out_for_oproj = NULL;

    if (is_full) {
        // ---- Full attention CPU compute ----
        int q_dim = cfg.num_attn_heads * cfg.head_dim;
        int kv_dim = cfg.num_kv_heads * cfg.head_dim;

        float *q = s_q;
        float *q_gate = s_q_gate;  // NULL for MiniMax (no gate)

        // Split q_proj_out into queries and gate (Qwen) or just copy (MiniMax)
        if (cfg.has_attn_gate) {
            for (int h = 0; h < cfg.num_attn_heads; h++) {
                float *src = q_proj_out + h * (2 * cfg.head_dim);
                memcpy(q + h * cfg.head_dim, src, cfg.head_dim * sizeof(float));
                memcpy(q_gate + h * cfg.head_dim, src + cfg.head_dim, cfg.head_dim * sizeof(float));
            }
        } else {
            memcpy(q, q_proj_out, q_dim * sizeof(float));
        }

        // Q/K RMSNorm
        uint16_t *qnorm_w = lc->q_norm_w;
        uint16_t *knorm_w = lc->k_norm_w;
        if (qnorm_w) {
            if (cfg.qk_norm_per_layer) {
                // MiniMax: RMSNorm over entire flat q vector [num_heads * head_dim]
                float sum_sq = 0.0f;
                for (int i = 0; i < q_dim; i++) sum_sq += q[i] * q[i];
                float inv_rms = 1.0f / sqrtf(sum_sq / q_dim + cfg.rms_norm_eps);
                for (int i = 0; i < q_dim; i++) q[i] = q[i] * inv_rms * bf16_to_f32(qnorm_w[i]);
            } else {
                // Qwen: RMSNorm per head, weight is [head_dim] shared across heads
                for (int h = 0; h < cfg.num_attn_heads; h++) {
                    float *qh = q + h * cfg.head_dim;
                    float sum_sq = 0.0f;
                    for (int i = 0; i < cfg.head_dim; i++) sum_sq += qh[i] * qh[i];
                    float inv_rms = 1.0f / sqrtf(sum_sq / cfg.head_dim + cfg.rms_norm_eps);
                    for (int i = 0; i < cfg.head_dim; i++) qh[i] = qh[i] * inv_rms * bf16_to_f32(qnorm_w[i]);
                }
            }
        }
        if (knorm_w) {
            if (cfg.qk_norm_per_layer) {
                // MiniMax: RMSNorm over entire flat k vector [num_kv_heads * head_dim]
                float sum_sq = 0.0f;
                for (int i = 0; i < kv_dim; i++) sum_sq += k_out[i] * k_out[i];
                float inv_rms = 1.0f / sqrtf(sum_sq / kv_dim + cfg.rms_norm_eps);
                for (int i = 0; i < kv_dim; i++) k_out[i] = k_out[i] * inv_rms * bf16_to_f32(knorm_w[i]);
            } else {
                // Qwen: RMSNorm per head
                for (int h = 0; h < cfg.num_kv_heads; h++) {
                    float *kh = k_out + h * cfg.head_dim;
                    float sum_sq = 0.0f;
                    for (int i = 0; i < cfg.head_dim; i++) sum_sq += kh[i] * kh[i];
                    float inv_rms = 1.0f / sqrtf(sum_sq / cfg.head_dim + cfg.rms_norm_eps);
                    for (int i = 0; i < cfg.head_dim; i++) kh[i] = kh[i] * inv_rms * bf16_to_f32(knorm_w[i]);
                }
            }
        }

        // RoPE
        apply_rotary_emb(q, k_out, pos, cfg.num_attn_heads, cfg.num_kv_heads, cfg.head_dim, cfg.rotary_dim);

        // Update KV cache (CPU + GPU mirror) — with sliding window + overflow protection
        int cache_pos;
        if (kv->h2o_active) {
            cache_pos = kv->h2o_num_valid;
            if (cache_pos >= kv->capacity) {
                fprintf(stderr, "ERROR: H2O KV cache overflow (pos=%d >= cap=%d)\n", cache_pos, kv->capacity);
                goto skip_full_attn;
            }
        } else if (kv->window_size > 0) {
            cache_pos = kv->len % kv->capacity;  // circular write
        } else {
            cache_pos = kv->len;
            if (cache_pos >= kv->capacity) {
                fprintf(stderr, "ERROR: KV cache overflow at layer %d (pos=%d >= cap=%d). "
                        "Consider --sliding-window or --h2o to bound context.\n",
                        layer_idx, cache_pos, kv->capacity);
                goto skip_full_attn;
            }
        }
        memcpy(kv->k_cache + cache_pos * kv_dim, k_out, kv_dim * sizeof(float));
        memcpy(kv->v_cache + cache_pos * kv_dim, v_out, kv_dim * sizeof(float));

        int fa_idx = cfg.full_attn_index[layer_idx];
        if (g_metal && g_metal->attn_scores_pipe && fa_idx >= 0 && fa_idx < cfg.num_full_attn_layers) {
            if (g_use_fp8_kv && g_metal->buf_kv_k_scales && g_metal->buf_kv_k_scales[fa_idx]) {
                // FP8 E4M3 encode: quantize K/V to uint8 with per-position scale
                uint8_t *k_fp8 = (uint8_t *)[g_metal->buf_kv_k[fa_idx] contents] + cache_pos * kv_dim;
                uint8_t *v_fp8 = (uint8_t *)[g_metal->buf_kv_v[fa_idx] contents] + cache_pos * kv_dim;
                float *k_scales = (float *)[g_metal->buf_kv_k_scales[fa_idx] contents];
                float *v_scales = (float *)[g_metal->buf_kv_v_scales[fa_idx] contents];
                k_scales[cache_pos] = fp8_encode_vec(k_out, k_fp8, kv_dim);
                v_scales[cache_pos] = fp8_encode_vec(v_out, v_fp8, kv_dim);
            } else {
                memcpy((float *)[g_metal->buf_kv_k[fa_idx] contents] + cache_pos * kv_dim,
                       k_out, kv_dim * sizeof(float));
                memcpy((float *)[g_metal->buf_kv_v[fa_idx] contents] + cache_pos * kv_dim,
                       v_out, kv_dim * sizeof(float));
            }
        }
        if (kv->h2o_active) {
            kv->token_positions[cache_pos] = kv->len;
            kv->attn_scores_accum[cache_pos] = 0.0f;
            kv->h2o_num_valid++;
        }
        kv->len++;

        // Scaled dot-product attention (GQA) — GPU or CPU
        int heads_per_kv = cfg.num_attn_heads / cfg.num_kv_heads;
        float scale = 1.0f / sqrtf((float)cfg.head_dim);
        float *attn_out = s_attn_out;
        memset(attn_out, 0, q_dim * sizeof(float));

        // Effective attention length (H2O, sliding window, or full)
        int attn_seq_len;
        if (kv->h2o_active) {
            attn_seq_len = kv->h2o_num_valid;
        } else if (kv->window_size > 0 && kv->len > kv->window_size) {
            attn_seq_len = kv->window_size;
        } else {
            attn_seq_len = kv->len;
        }

        // GPU attention: defer dispatches to CMD2 (fused into single cmd buffer).
        // Only enabled when seq_len >= 32 (below that, CPU is faster).
        int gpu_attn_ready = (g_metal && g_metal->attn_scores_pipe &&
                              fa_idx >= 0 && fa_idx < cfg.num_full_attn_layers &&
                              attn_seq_len >= 32 && attn_seq_len < GPU_KV_SEQ &&
                              !kv->h2o_active);  // H2O uses CPU for score accumulation

        if (gpu_attn_ready) {
            // Copy Q and gate to GPU; attention dispatches will be in CMD2
            memcpy([g_metal->buf_attn_q contents], q, q_dim * sizeof(float));
            if (q_gate)
                memcpy([g_metal->buf_attn_gate contents], q_gate, q_dim * sizeof(float));
            else
                memset([g_metal->buf_attn_gate contents], 0, q_dim * sizeof(float)); // no gate = no-op sigmoid(0)=0.5 — but GPU sigmoid_gate kernel will be skipped
            // attn_out_for_oproj will be set to NULL below — CMD2 reads buf_attn_out
        } else {
            // CPU fallback (also used when H2O is active, for score tracking)
            for (int h = 0; h < cfg.num_attn_heads; h++) {
                int kv_h = h / heads_per_kv;
                float *qh = q + h * cfg.head_dim;
                float *scores = malloc(attn_seq_len * sizeof(float));
                if (!scores) continue;
                for (int i = 0; i < attn_seq_len; i++) {
                    int p;
                    if (!kv->h2o_active && kv->window_size > 0 && kv->len > kv->window_size) {
                        p = (kv->len - kv->window_size + i) % kv->capacity;
                    } else {
                        p = i;
                    }
                    float *kp = kv->k_cache + p * kv_dim + kv_h * cfg.head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < cfg.head_dim; d++) dot += qh[d] * kp[d];
                    scores[i] = dot * scale;
                }
                cpu_softmax(scores, attn_seq_len);
                float *oh = attn_out + h * cfg.head_dim;
                for (int i = 0; i < attn_seq_len; i++) {
                    int p;
                    if (!kv->h2o_active && kv->window_size > 0 && kv->len > kv->window_size) {
                        p = (kv->len - kv->window_size + i) % kv->capacity;
                    } else {
                        p = i;
                    }
                    float *vp = kv->v_cache + p * kv_dim + kv_h * cfg.head_dim;
                    for (int d = 0; d < cfg.head_dim; d++) oh[d] += scores[i] * vp[d];
                }
                free(scores);
            }
            // Apply sigmoid gate (Qwen only)
            if (q_gate) {
                for (int i = 0; i < q_dim; i++) {
                    float g = 1.0f / (1.0f + expf(-q_gate[i]));
                    attn_out[i] *= g;
                }
            }
        }

        if (gpu_attn_ready) {
            attn_out_for_oproj = NULL;  // signal CMD2 to use GPU buf_attn_out
        } else {
            attn_out_for_oproj = attn_out;
        }
        // q_proj_out, k_out, v_out, q, q_gate, attn_out are static scratch.
        skip_full_attn:; // jump target when KV cache overflows
    } else if (gpu_linear_attn) {
        // ---- GPU linear attention: already computed in CMD1 ----
        // batch_out[6] already contains gated_rms_norm output (8192 floats)
        // Set a non-NULL sentinel so CMD2 enters fused path, but skip the memcpy
        static float gpu_linear_sentinel;
        attn_out_for_oproj = &gpu_linear_sentinel;
    } else {
        // ---- Linear attention CPU compute ----
        if (!linear_attn_bypass) {
            int qkv_dim = cfg.linear_conv_dim;

            // Conv1d step
            uint16_t *conv_w = lc->conv1d_w;
            float *conv_out = s_conv_out;
            memset(conv_out, 0, qkv_dim * sizeof(float));
            if (conv_w) {
                cpu_conv1d_step(la_state->conv_state, qkv_out, conv_w, conv_out,
                                qkv_dim, cfg.conv_kernel_size);
            }
            // Update conv state
            memmove(la_state->conv_state, la_state->conv_state + qkv_dim,
                    (cfg.conv_kernel_size - 2) * qkv_dim * sizeof(float));
            memcpy(la_state->conv_state + (cfg.conv_kernel_size - 2) * qkv_dim, qkv_out,
                   qkv_dim * sizeof(float));

            // Split into q, k, v
            float *lin_q = conv_out;
            float *lin_k = conv_out + cfg.linear_total_key;
            float *lin_v = conv_out + 2 * cfg.linear_total_key;

            // RMS normalize q and k
            float inv_scale = 1.0f / sqrtf((float)cfg.linear_key_dim);
            for (int h = 0; h < cfg.linear_num_k_heads; h++) {
                float *qh = lin_q + h * cfg.linear_key_dim;
                cpu_rms_norm_bare(qh, qh, cfg.linear_key_dim, 1e-6f);
                float q_scale = inv_scale * inv_scale;
                for (int d = 0; d < cfg.linear_key_dim; d++) qh[d] *= q_scale;
            }
            for (int h = 0; h < cfg.linear_num_k_heads; h++) {
                float *kh = lin_k + h * cfg.linear_key_dim;
                cpu_rms_norm_bare(kh, kh, cfg.linear_key_dim, 1e-6f);
                for (int d = 0; d < cfg.linear_key_dim; d++) kh[d] *= inv_scale;
            }

            // Gated delta net recurrence
            float *A_log = lc->A_log;
            uint16_t *dt_bias_bf16 = lc->dt_bias;

            float *out_values = s_out_vals;
            memset(out_values, 0, cfg.linear_total_value * sizeof(float));
            int k_heads_per_v = cfg.linear_num_v_heads / cfg.linear_num_k_heads;

            float g_decay[cfg.linear_num_v_heads];
            float beta_gate_arr[cfg.linear_num_v_heads];
            for (int vh = 0; vh < cfg.linear_num_v_heads; vh++) {
                float a_val = alpha_out[vh];
                float dt_b = dt_bias_bf16 ? bf16_to_f32(dt_bias_bf16[vh]) : 0.0f;
                float A_val = A_log ? expf(A_log[vh]) : 1.0f;
                float softplus_val = logf(1.0f + expf(a_val + dt_b));
                g_decay[vh] = expf(-A_val * softplus_val);
                beta_gate_arr[vh] = cpu_sigmoid(beta_out[vh]);
            }

            // Compute linear_layer_idx: count of non-full-attention layers before this one.
            // Full attention at (layer_idx+1) % 4 == 0, i.e. layers 3,7,11,...
            // linear_layer_idx = layer_idx - number_of_full_layers_at_or_before
            //                  = cfg.linear_index[layer_idx]
            int linear_layer_idx = cfg.linear_index[layer_idx];

            // GPU delta-net path (falls back to CPU if pipeline unavailable)
            if (g_metal && g_metal->delta_net_step &&
                linear_layer_idx >= 0 && linear_layer_idx < cfg.num_linear_layers) {
                // Upload CPU-computed data to GPU scratch buffers
                memcpy([g_metal->buf_delta_q contents], lin_q, cfg.linear_total_key * sizeof(float));
                memcpy([g_metal->buf_delta_k contents], lin_k, cfg.linear_total_key * sizeof(float));
                memcpy([g_metal->buf_delta_v contents], lin_v, cfg.linear_total_value * sizeof(float));
                memcpy([g_metal->buf_delta_g_decay contents], g_decay, cfg.linear_num_v_heads * sizeof(float));
                memcpy([g_metal->buf_delta_beta contents], beta_gate_arr, cfg.linear_num_v_heads * sizeof(float));

                id<MTLCommandBuffer> cmd_dn = [g_metal->queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd_dn computeCommandEncoder];
                [enc setComputePipelineState:g_metal->delta_net_step_fused ? g_metal->delta_net_step_fused : g_metal->delta_net_step];
                [enc setBuffer:g_metal->buf_delta_state[linear_layer_idx] offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_delta_q       offset:0 atIndex:1];
                [enc setBuffer:g_metal->buf_delta_k       offset:0 atIndex:2];
                [enc setBuffer:g_metal->buf_delta_v       offset:0 atIndex:3];
                [enc setBuffer:g_metal->buf_delta_g_decay offset:0 atIndex:4];
                [enc setBuffer:g_metal->buf_delta_beta    offset:0 atIndex:5];
                [enc setBuffer:g_metal->buf_delta_output  offset:0 atIndex:6];
                uint32_t khpv = (uint32_t)k_heads_per_v;
                [enc setBytes:&khpv length:sizeof(khpv) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(cfg.linear_num_v_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding];
                [cmd_dn commit];
                [cmd_dn waitUntilCompleted];

                // Read back GPU result
                memcpy(out_values, [g_metal->buf_delta_output contents], cfg.linear_total_value * sizeof(float));
            } else {
                // CPU delta-net with Accelerate BLAS
                for (int vh = 0; vh < cfg.linear_num_v_heads; vh++) {
                    int kh = vh / k_heads_per_v;
                    float g = g_decay[vh];
                    float b_gate = beta_gate_arr[vh];
                    float *S = la_state->ssm_state + vh * cfg.linear_value_dim * cfg.linear_key_dim;
                    float *v_h = lin_v + vh * cfg.linear_value_dim;
                    float *k_h = lin_k + kh * cfg.linear_key_dim;

                    // Step 1: Decay S *= g (BLAS sscal on entire state matrix)
                    cblas_sscal(cfg.linear_value_dim * cfg.linear_key_dim, g, S, 1);

                    // Step 2: kv_mem = S @ k (each row dot k)
                    // S is [VALUE_DIM x KEY_DIM] row-major, k is [KEY_DIM]
                    // kv_mem[vi] = sum_ki(S[vi,ki] * k[ki]) = matrix-vector: S @ k
                    float kv_mem_vec[cfg.linear_value_dim];
                    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                                cfg.linear_value_dim, cfg.linear_key_dim,
                                1.0f, S, cfg.linear_key_dim, k_h, 1,
                                0.0f, kv_mem_vec, 1);

                    // Step 3: delta = (v - kv_mem) * beta, then rank-1 update S += k * delta^T
                    // delta[vi] = (v[vi] - kv_mem[vi]) * beta
                    float delta_vec[cfg.linear_value_dim];
                    for (int vi = 0; vi < cfg.linear_value_dim; vi++) {
                        delta_vec[vi] = (v_h[vi] - kv_mem_vec[vi]) * b_gate;
                    }
                    // S += delta @ k^T (rank-1 update: sger)
                    // S[vi,ki] += delta[vi] * k[ki]
                    cblas_sger(CblasRowMajor, cfg.linear_value_dim, cfg.linear_key_dim,
                               1.0f, delta_vec, 1, k_h, 1, S, cfg.linear_key_dim);

                    // Step 4: output = S @ q (matrix-vector multiply)
                    float *q_h = lin_q + kh * cfg.linear_key_dim;
                    float *o_h = out_values + vh * cfg.linear_value_dim;
                    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                                cfg.linear_value_dim, cfg.linear_key_dim,
                                1.0f, S, cfg.linear_key_dim, q_h, 1,
                                0.0f, o_h, 1);
                }
            }

            // RMSNormGated
            uint16_t *gated_norm_w = lc->gated_norm_w;
            float *gated_out = s_gated_out;
            memset(gated_out, 0, cfg.linear_total_value * sizeof(float));
            for (int vh = 0; vh < cfg.linear_num_v_heads; vh++) {
                float *oh = out_values + vh * cfg.linear_value_dim;
                float *zh = z_out + vh * cfg.linear_value_dim;
                float *gh = gated_out + vh * cfg.linear_value_dim;
                if (gated_norm_w) {
                    cpu_rms_norm_gated(oh, zh, gated_norm_w, gh, cfg.linear_value_dim, cfg.rms_norm_eps);
                } else {
                    memcpy(gh, oh, cfg.linear_value_dim * sizeof(float));
                }
            }

            attn_out_for_oproj = gated_out;

            // conv_out, out_values are static — no free needed
            // gated_out is static — freed/released after CMD2 submission below
        }
        // else: linear_attn_bypass — attn_projected stays zero
        // qkv_out, z_out, beta_out, alpha_out are static scratch.
    }

    // =====================================================================
    // PHASE 3: FULLY FUSED CMD2 — o_proj + residual + norm + routing (1 cmd buffer)
    //   Eliminates 1 GPU round-trip vs old 2-buffer approach.
    //   GPU handles residual_add + rms_norm between o_proj and routing,
    //   so no CPU intervention is needed. 8 encoders, 1 commit+wait.
    //   Buffer flow: batch_out[6]->buf_output->buf_h_mid->buf_input->batch_out[0-3]
    // =====================================================================

    if (g_timing_enabled) { t1 = now_ms(); g_timing.cpu_attn += t1 - t0; }

    // Wait for speculative expert I/O to complete (overlapped with CPU attention)
    if (spec_group) {
        dispatch_group_wait(spec_group, DISPATCH_TIME_FOREVER);
        spec_group = NULL;  // ARC releases the group
    }

    if (g_timing_enabled) { t0 = now_ms(); }

    float *h_post = s_h_post;
    float *h_mid = s_h_mid;
    float *gate_scores = s_gate_scores;
    float *shared_gate = s_shared_gate;
    float *shared_up = s_shared_up;
    float shared_gate_score = 0.0f;

    // CMD1+CMD2 merged path: gate_scores/shared_gate/shared_up are already
    // populated by gpu_flush_batch_results — skip directly to softmax+topK.
    // MUST jump before memset, which would zero out the GPU results.
    if (cmd1_cmd2_merged && !is_full) {
        shared_gate_score = g_merged_shared_gate_score;
        // h_mid and h_post already populated from GPU readback
        cpu_vec_copy(residual, h_mid, cfg.hidden_dim);
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd2_wait += t1 - t0; }
        goto cmd2_done;
    }

    // Zero-init for non-merged path (CMD2 will populate these)
    memset(gate_scores, 0, cfg.num_experts * sizeof(float));
    if (cfg.shared_intermediate > 0) {
        memset(shared_gate, 0, cfg.shared_intermediate * sizeof(float));
        memset(shared_up, 0, cfg.shared_intermediate * sizeof(float));
    }

    int have_moe_weights = (gate_w && gate_s && gate_b);
    int have_shared_weights = (cfg.shared_intermediate > 0 && sgw && sgs && sgb &&
                               suw && sus && sub && seg_w && seg_s && seg_b);

    // gpu_attn_fuse: attention dispatches fused into CMD2 (full-attn layers only).
    // Only enabled when seq_len >= 32 — below that, CPU attention is faster
    // because GPU command encoder overhead dominates at short sequences.
    // Use attn_seq_len for GPU attention when available (respects sliding window / H2O)
    int fused_attn_len = (is_full && kv) ?
        (kv->h2o_active ? kv->h2o_num_valid :
         (kv->window_size > 0 && kv->len > kv->window_size) ? kv->window_size : kv->len) : 0;
    int gpu_attn_fuse = (is_full && !attn_out_for_oproj && g_metal && g_metal->attn_scores_pipe
                         && kv && fused_attn_len >= 32 && fused_attn_len < GPU_KV_SEQ
                         && !kv->h2o_active);

    if ((attn_out_for_oproj || gpu_attn_fuse) && oproj_w && oproj_s && oproj_b &&
        g_metal && g_metal->wf_buf && have_moe_weights &&
        g_metal->residual_add && g_metal->rms_norm_sum &&
        g_metal->rms_norm_apply_bf16 && lc->post_attn_norm_w) {
        // ---- FULLY FUSED CMD2 ----
        // For GPU attention (full-attn layers): attention dispatches are prepended,
        //   o_proj reads from buf_attn_out instead of batch_out[6].
        // For CPU attention / linear attn: o_proj reads from batch_out[6] as before.
        //
        // GPU attn path (12 encoders):
        //   Enc 1-4: attn_scores + softmax + values + sigmoid -> buf_attn_out
        //   Enc 5:   o_proj (buf_attn_out -> buf_output)
        //   Enc 6-8: residual + norm -> buf_input
        //   Enc 9-12: routing + shared expert
        //
        // CPU attn path (8 encoders, unchanged):
        //   Enc 1:   o_proj (batch_out[6] -> buf_output)
        //   Enc 2-4: residual + norm -> buf_input
        //   Enc 5-8: routing + shared expert

        if (!gpu_attn_fuse && !gpu_linear_attn) {
            // CPU/linear attn: copy attention output to GPU input buffer
            memcpy([g_metal->batch_out[6] contents], attn_out_for_oproj,
                   oproj_in_dim * sizeof(float));
        }
        // gpu_linear_attn: batch_out[6] already has the result from CMD1 gated_rms_norm
        // Copy residual into GPU buffer for residual_add kernel
        memcpy([g_metal->buf_residual contents], residual, cfg.hidden_dim * sizeof(float));

        attn_out_for_oproj = NULL;

        id<MTLCommandBuffer> cmd_fused = [g_metal->queue commandBuffer];

        // ---- GPU attention dispatches (only for full-attn layers with GPU path) ----
        if (gpu_attn_fuse) {
            int fa_idx = cfg.full_attn_index[layer_idx];
            int kv_dim = cfg.num_kv_heads * cfg.head_dim;
            int heads_per_kv = cfg.num_attn_heads / cfg.num_kv_heads;
            float scale = 1.0f / sqrtf((float)cfg.head_dim);
            uint32_t hd = cfg.head_dim;
            uint32_t kvd = (uint32_t)kv_dim;
            uint32_t sl = (uint32_t)fused_attn_len;
            uint32_t seq_stride = GPU_KV_SEQ;
            uint32_t hpkv = (uint32_t)heads_per_kv;
            uint32_t num_heads = cfg.num_attn_heads;
            uint32_t num_kv_heads = cfg.num_kv_heads;

            // Fused online softmax attention: single kernel replaces 3-kernel pipeline
            if (g_fused_attention_enabled && g_metal->fused_attention_fc_pipe) {
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc setComputePipelineState:g_metal->fused_attention_fc_pipe];
                [enc setBuffer:g_metal->buf_attn_q          offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_kv_k[fa_idx]    offset:0 atIndex:1];
                if (g_use_fp8_kv && g_metal->buf_kv_k_scales && g_metal->buf_kv_k_scales[fa_idx]) {
                    [enc setBuffer:g_metal->buf_kv_k_scales[fa_idx] offset:0 atIndex:2];
                } else {
                    [enc setBuffer:g_metal->buf_attn_scores  offset:0 atIndex:2]; // unused placeholder
                }
                [enc setBuffer:g_metal->buf_kv_v[fa_idx]    offset:0 atIndex:3];
                if (g_use_fp8_kv && g_metal->buf_kv_v_scales && g_metal->buf_kv_v_scales[fa_idx]) {
                    [enc setBuffer:g_metal->buf_kv_v_scales[fa_idx] offset:0 atIndex:4];
                } else {
                    [enc setBuffer:g_metal->buf_attn_scores  offset:0 atIndex:4]; // unused placeholder
                }
                [enc setBuffer:g_metal->buf_attn_out        offset:0 atIndex:5];
                [enc setBytes:&hd           length:4 atIndex:6];
                [enc setBytes:&kvd          length:4 atIndex:7];
                [enc setBytes:&sl           length:4 atIndex:8];
                [enc setBytes:&num_heads    length:4 atIndex:9];
                [enc setBytes:&num_kv_heads length:4 atIndex:10];
                [enc setBytes:&scale        length:4 atIndex:11];
                [enc dispatchThreadgroups:MTLSizeMake(cfg.num_attn_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            } else if (g_use_fp8_kv && g_metal->attn_scores_fp8_pipe && g_metal->attn_values_fp8_pipe &&
                       g_metal->buf_kv_k_scales && g_metal->buf_kv_k_scales[fa_idx]) {
                // FP8 3-kernel pipeline
                // Enc A1: attn_scores_fp8
                {
                    id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->attn_scores_fp8_pipe];
                    [enc setBuffer:g_metal->buf_attn_q              offset:0 atIndex:0];
                    [enc setBuffer:g_metal->buf_kv_k[fa_idx]        offset:0 atIndex:1];
                    [enc setBuffer:g_metal->buf_kv_k_scales[fa_idx] offset:0 atIndex:2];
                    [enc setBuffer:g_metal->buf_attn_scores         offset:0 atIndex:3];
                    [enc setBytes:&hd        length:4 atIndex:4];
                    [enc setBytes:&kvd       length:4 atIndex:5];
                    [enc setBytes:&sl        length:4 atIndex:6];
                    [enc setBytes:&seq_stride length:4 atIndex:7];
                    [enc setBytes:&scale     length:4 atIndex:8];
                    [enc setBytes:&hpkv      length:4 atIndex:9];
                    [enc setBytes:&sl        length:4 atIndex:10];
                    uint32_t total_tgs = sl * cfg.num_attn_heads;
                    [enc dispatchThreadgroups:MTLSizeMake(total_tgs, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                }
                // Enc A2: attn_softmax_batched (same as float32 path)
                {
                    id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->attn_softmax_pipe];
                    [enc setBuffer:g_metal->buf_attn_scores offset:0 atIndex:0];
                    [enc setBytes:&sl         length:4 atIndex:1];
                    [enc setBytes:&seq_stride  length:4 atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake(cfg.num_attn_heads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                }
                // Enc A3: attn_values_fp8
                {
                    id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->attn_values_fp8_pipe];
                    [enc setBuffer:g_metal->buf_attn_scores         offset:0 atIndex:0];
                    [enc setBuffer:g_metal->buf_kv_v[fa_idx]        offset:0 atIndex:1];
                    [enc setBuffer:g_metal->buf_kv_v_scales[fa_idx] offset:0 atIndex:2];
                    [enc setBuffer:g_metal->buf_attn_out            offset:0 atIndex:3];
                    [enc setBytes:&hd        length:4 atIndex:4];
                    [enc setBytes:&kvd       length:4 atIndex:5];
                    [enc setBytes:&sl        length:4 atIndex:6];
                    [enc setBytes:&seq_stride length:4 atIndex:7];
                    [enc setBytes:&hpkv      length:4 atIndex:8];
                    uint32_t total_threads = cfg.head_dim * cfg.num_attn_heads;
                    uint32_t tgs = (total_threads + 255) / 256;
                    [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                }
            } else {
                // Standard float32 3-kernel pipeline
                // Enc A1: attn_scores_batched
                {
                    id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->attn_scores_pipe];
                    [enc setBuffer:g_metal->buf_attn_q          offset:0 atIndex:0];
                    [enc setBuffer:g_metal->buf_kv_k[fa_idx]    offset:0 atIndex:1];
                    [enc setBuffer:g_metal->buf_attn_scores     offset:0 atIndex:2];
                    [enc setBytes:&hd        length:4 atIndex:3];
                    [enc setBytes:&kvd       length:4 atIndex:4];
                    [enc setBytes:&sl        length:4 atIndex:5];
                    [enc setBytes:&seq_stride length:4 atIndex:6];
                    [enc setBytes:&scale     length:4 atIndex:7];
                    [enc setBytes:&hpkv      length:4 atIndex:8];
                    [enc setBytes:&sl        length:4 atIndex:9];
                    uint32_t total_tgs = sl * cfg.num_attn_heads;
                    [enc dispatchThreadgroups:MTLSizeMake(total_tgs, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                }
                // Enc A2: attn_softmax_batched
                {
                    id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->attn_softmax_pipe];
                    [enc setBuffer:g_metal->buf_attn_scores offset:0 atIndex:0];
                    [enc setBytes:&sl         length:4 atIndex:1];
                    [enc setBytes:&seq_stride  length:4 atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake(cfg.num_attn_heads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                }
                // Enc A3: attn_values_batched
                {
                    id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->attn_values_pipe];
                    [enc setBuffer:g_metal->buf_attn_scores   offset:0 atIndex:0];
                    [enc setBuffer:g_metal->buf_kv_v[fa_idx]  offset:0 atIndex:1];
                    [enc setBuffer:g_metal->buf_attn_out      offset:0 atIndex:2];
                    [enc setBytes:&hd        length:4 atIndex:3];
                    [enc setBytes:&kvd       length:4 atIndex:4];
                    [enc setBytes:&sl        length:4 atIndex:5];
                    [enc setBytes:&seq_stride length:4 atIndex:6];
                    [enc setBytes:&hpkv      length:4 atIndex:7];
                    uint32_t total_threads = cfg.head_dim * cfg.num_attn_heads;
                    uint32_t tgs = (total_threads + 255) / 256;
                    [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                }
            }
            // Enc A4: sigmoid_gate (Qwen only — skip for MiniMax)
            if (cfg.has_attn_gate) {
                uint32_t qdim = cfg.num_attn_heads * cfg.head_dim;
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc setComputePipelineState:g_metal->sigmoid_gate_pipe];
                [enc setBuffer:g_metal->buf_attn_out  offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_attn_gate offset:0 atIndex:1];
                [enc setBytes:&qdim length:4 atIndex:2];
                uint32_t tgs = (qdim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
        }

        // ---- o_proj matvec ----
        {
            // For GPU attention: o_proj reads from buf_attn_out
            // For CPU attention: o_proj reads from batch_out[6]
            id<MTLBuffer> oproj_input = gpu_attn_fuse ? g_metal->buf_attn_out : g_metal->batch_out[6];

            uint32_t o_out_dim = cfg.hidden_dim;
            uint32_t o_in_dim = (uint32_t)oproj_in_dim;
            uint32_t o_gs = cfg.group_size;
            id<MTLBuffer> ow_buf, os_buf, ob_buf;
            NSUInteger ow_off, os_off, ob_off;
            size_t oproj_w_size = (size_t)o_out_dim * o_in_dim / 8;  // 4-bit packed
            size_t oproj_ng = (o_in_dim + o_gs - 1) / o_gs;
            size_t oproj_sb_size = (size_t)o_out_dim * oproj_ng * sizeof(uint16_t);
            metal_staging_reset(g_metal);
            metal_find_chunk_sized(g_metal, oproj_w, oproj_w_size, &ow_buf, &ow_off);
            metal_find_chunk_sized(g_metal, oproj_s, oproj_sb_size, &os_buf, &os_off);
            metal_find_chunk_sized(g_metal, oproj_b, oproj_sb_size, &ob_buf, &ob_off);
            id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
            [enc setComputePipelineState:g_metal->matvec_fast];
            [enc setBuffer:ow_buf offset:ow_off atIndex:0];
            [enc setBuffer:os_buf offset:os_off atIndex:1];
            [enc setBuffer:ob_buf offset:ob_off atIndex:2];
            [enc setBuffer:oproj_input      offset:0    atIndex:3];
            [enc setBuffer:g_metal->buf_output offset:0 atIndex:4];
            [enc setBytes:&o_out_dim  length:4 atIndex:5];
            [enc setBytes:&o_in_dim   length:4 atIndex:6];
            [enc setBytes:&o_gs       length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(o_out_dim, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
        }

        // ---- Enc 2: residual_add (buf_output + buf_residual -> buf_h_mid) ----
        {
            id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
            uint32_t dim = cfg.hidden_dim;
            [enc setComputePipelineState:g_metal->residual_add];
            [enc setBuffer:g_metal->buf_residual offset:0 atIndex:0];  // a = residual
            [enc setBuffer:g_metal->buf_output   offset:0 atIndex:1];  // b = o_proj result
            [enc setBuffer:g_metal->buf_h_mid    offset:0 atIndex:2];  // out = h_mid
            [enc setBytes:&dim length:4 atIndex:3];
            uint32_t tgs = (dim + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // ---- Enc 3: rms_norm_sum_sq (buf_h_mid -> buf_sum_sq) ----
        {
            id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
            uint32_t dim = cfg.hidden_dim;
            [enc setComputePipelineState:g_metal->rms_norm_sum];
            [enc setBuffer:g_metal->buf_h_mid  offset:0 atIndex:0];
            [enc setBuffer:g_metal->buf_sum_sq offset:0 atIndex:1];
            [enc setBytes:&dim length:4 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // ---- Enc 4: rms_norm_apply_bf16 (buf_h_mid + norm_w -> buf_input) ----
        {
            id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
            id<MTLBuffer> panw_buf; NSUInteger panw_off;
            // Don't reset staging — o_proj data is still being read by earlier encoders in cmd_fused
            metal_find_chunk_sized(g_metal, lc->post_attn_norm_w,
                (size_t)cfg.hidden_dim * 2, &panw_buf, &panw_off);  // bf16
            uint32_t dim = cfg.hidden_dim;
            float eps = cfg.rms_norm_eps;
            [enc setComputePipelineState:g_metal->rms_norm_apply_bf16];
            [enc setBuffer:g_metal->buf_h_mid  offset:0       atIndex:0];  // x
            [enc setBuffer:panw_buf offset:panw_off atIndex:1]; // weight (bf16)
            [enc setBuffer:g_metal->buf_sum_sq offset:0       atIndex:2];  // sum_sq
            [enc setBuffer:g_metal->buf_input  offset:0       atIndex:3];  // out = h_post
            [enc setBytes:&dim length:4 atIndex:4];
            [enc setBytes:&eps length:4 atIndex:5];
            uint32_t tgs = (dim + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // ---- Enc 5+: routing gate + (optionally) shared expert projections ----
        // If routing gate uses different quantization (e.g. 8-bit for MiniMax),
        // skip it from GPU batch and compute on CPU after readback.
        int gate_on_cpu = (cfg.gate_bits != cfg.bits);
        int num_moe_specs;
        BatchMatvecSpec moe_specs[4];
        if (gate_on_cpu) {
            // GPU: only shared expert specs (if any), skip gate
            if (have_shared_weights) {
                moe_specs[0] = (BatchMatvecSpec){ sgw,    sgs,    sgb,    shared_gate,         (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 1 };
                moe_specs[1] = (BatchMatvecSpec){ suw,    sus,    sub,    shared_up,           (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 2 };
                moe_specs[2] = (BatchMatvecSpec){ seg_w,  seg_s,  seg_b,  &shared_gate_score,  1,                            cfg.hidden_dim, cfg.group_size, 3 };
                num_moe_specs = 3;
            } else {
                num_moe_specs = 0;  // Nothing to batch on GPU
            }
        } else if (have_shared_weights) {
            moe_specs[0] = (BatchMatvecSpec){ gate_w, gate_s, gate_b, gate_scores,        (uint32_t)cfg.num_experts,        cfg.hidden_dim, cfg.group_size, 0 };
            moe_specs[1] = (BatchMatvecSpec){ sgw,    sgs,    sgb,    shared_gate,         (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 1 };
            moe_specs[2] = (BatchMatvecSpec){ suw,    sus,    sub,    shared_up,           (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 2 };
            moe_specs[3] = (BatchMatvecSpec){ seg_w,  seg_s,  seg_b,  &shared_gate_score,  1,                            cfg.hidden_dim, cfg.group_size, 3 };
            num_moe_specs = 4;
        } else {
            moe_specs[0] = (BatchMatvecSpec){ gate_w, gate_s, gate_b, gate_scores, (uint32_t)cfg.num_experts, cfg.hidden_dim, cfg.group_size, 0 };
            num_moe_specs = 1;
        }
        // buf_input already contains h_post from Enc 4 output -- no memcpy needed
        if (num_moe_specs > 0)
            gpu_encode_batch_matvec(g_metal, cmd_fused, moe_specs, num_moe_specs);

        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd2_encode += t1 - t0; }

        // ---- Single commit+wait ----
        if (g_timing_enabled) { t0 = now_ms(); }
        [cmd_fused commit];
        [cmd_fused waitUntilCompleted];

        // Read back results
        if (num_moe_specs > 0)
            gpu_flush_batch_results(g_metal, moe_specs, num_moe_specs);
        // Read h_mid from GPU buffer (needed for final combine)
        memcpy(h_mid, [g_metal->buf_h_mid contents], cfg.hidden_dim * sizeof(float));
        // Read h_post from buf_input (needed for expert input)
        memcpy(h_post, [g_metal->buf_input contents], cfg.hidden_dim * sizeof(float));
        // Update hidden state to h_mid (= residual + o_proj)
        memcpy(hidden, h_mid, cfg.hidden_dim * sizeof(float));
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd2_wait += t1 - t0; }

        // CPU routing gate for non-standard quantization (e.g. 8-bit MiniMax gate)
        if (gate_on_cpu && gate_w && gate_s && gate_b) {
            cpu_dequant_matvec_nbits((const uint32_t *)gate_w, (const uint16_t *)gate_s, (const uint16_t *)gate_b,
                                     h_post, gate_scores,
                                     cfg.num_experts, cfg.hidden_dim, cfg.gate_group_size, cfg.gate_bits);
        }

    } else {
        // ---- Non-fused fallback path ----
        // O projection
        if (attn_out_for_oproj && oproj_w && oproj_s && oproj_b) {
            fast_dequant_matvec(oproj_w, oproj_s, oproj_b, attn_out_for_oproj,
                                attn_projected, cfg.hidden_dim, oproj_in_dim, cfg.group_size);
        }
        // attn_out_for_oproj is static — no free needed
        attn_out_for_oproj = NULL;

        // Residual connection
        for (int i = 0; i < cfg.hidden_dim; i++) {
            hidden[i] = residual[i] + attn_projected[i];
        }
        // attn_projected, normed, residual are static — no free needed

        cpu_vec_copy(h_mid, hidden, cfg.hidden_dim);

        // Post-attention norm
        cpu_rms_norm(hidden, lc->post_attn_norm_w, h_post, cfg.hidden_dim, cfg.rms_norm_eps);

        // Routing + (optionally) shared expert batch
        if (have_moe_weights) {
            // If routing gate uses different quantization (e.g. 8-bit), compute it separately on CPU
            if (cfg.gate_bits != cfg.bits) {
                cpu_dequant_matvec_nbits(gate_w, gate_s, gate_b, h_post, gate_scores,
                                         cfg.num_experts, cfg.hidden_dim, cfg.gate_group_size, cfg.gate_bits);
                // Batch only shared expert specs (if any)
                if (have_shared_weights) {
                    BatchMatvecSpec moe_specs[3] = {
                        { sgw,    sgs,    sgb,    shared_gate,         (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 1 },
                        { suw,    sus,    sub,    shared_up,           (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 2 },
                        { seg_w,  seg_s,  seg_b,  &shared_gate_score,  1,                            cfg.hidden_dim, cfg.group_size, 3 },
                    };
                    fast_batch_matvec(h_post, cfg.hidden_dim, moe_specs, 3);
                }
            } else if (have_shared_weights) {
                BatchMatvecSpec moe_specs[4] = {
                    { gate_w, gate_s, gate_b, gate_scores,        (uint32_t)cfg.num_experts,        cfg.hidden_dim, cfg.group_size, 0 },
                    { sgw,    sgs,    sgb,    shared_gate,         (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 1 },
                    { suw,    sus,    sub,    shared_up,           (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 2 },
                    { seg_w,  seg_s,  seg_b,  &shared_gate_score,  1,                            cfg.hidden_dim, cfg.group_size, 3 },
                };
                fast_batch_matvec(h_post, cfg.hidden_dim, moe_specs, 4);
            } else {
                BatchMatvecSpec moe_specs[1] = {
                    { gate_w, gate_s, gate_b, gate_scores, (uint32_t)cfg.num_experts, cfg.hidden_dim, cfg.group_size, 0 },
                };
                fast_batch_matvec(h_post, cfg.hidden_dim, moe_specs, 1);
            }
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd2_encode += t1 - t0; }
    }

cmd2_done:
    // ---- Route experts (CPU) ----
    if (g_timing_enabled) { t0 = now_ms(); }
    int expert_indices[64];
    float expert_weights[64];
    cpu_route_experts(gate_scores, cfg.num_experts, K,
                      lc->routing_bias, expert_indices, expert_weights);
    if (g_freq_tracking) {
        for (int k = 0; k < K; k++) {
            FREQ(layer_idx, expert_indices[k])++;
        }
        if (layer_idx == 0) g_freq_total_tokens++;
    }

    // Track speculative routing prediction accuracy
    if (s_spec_count > 0) {
        int cmp_K = (K > MAX_K) ? MAX_K : K;
        for (int s = 0; s < s_spec_count; s++) {
            for (int r = 0; r < cmp_K; r++) {
                if (s_spec_indices[s] == expert_indices[r]) {
                    g_spec_route_hits++;
                    break;
                }
            }
        }
    }

    if (g_timing_enabled) { t1 = now_ms(); g_timing.routing_cpu += t1 - t0; }

    // Log routing data for predictor training
    if (g_routing_log) {
        int32_t li = layer_idx;
        int32_t ki = (K > MAX_K) ? MAX_K : K;
        fwrite(&li, sizeof(int32_t), 1, g_routing_log);
        fwrite(&ki, sizeof(int32_t), 1, g_routing_log);
        fwrite(hidden, sizeof(float), cfg.hidden_dim, g_routing_log);
        fwrite(expert_indices, sizeof(int32_t), ki, g_routing_log);
        g_routing_log_samples++;
    }

    // ---- Parallel pread + GPU experts ----
    if (g_timing_enabled) { t0 = now_ms(); }
    float *moe_out = s_moe_out;
    memset(moe_out, 0, cfg.hidden_dim * sizeof(float));
    float *shared_out = s_shared_out;
    memset(shared_out, 0, cfg.hidden_dim * sizeof(float));

    int actual_K = (K > MAX_K) ? MAX_K : K;

    if (packed_fd >= 0 && g_metal && g_metal->buf_multi_expert_data[0]) {
        // GPU multi-expert path with LRU cache + parallel I/O:
        // For each expert:
        //   - Cache HIT:  dispatch directly from cached Metal buffer (skip pread)
        //   - Cache MISS: pread into cache buffer, then dispatch from it
        // Falls back to original parallel_pread_experts when cache is disabled.

        int valid[MAX_K];
        id<MTLBuffer> expert_bufs[MAX_K];  // buffer to dispatch from per expert

        if (g_malloc_cache) {
            // ---- Malloc cache path (zero-copy Metal buffer wrappers) ----
            // Phase 1: check cache for each expert, collect misses
            int miss_indices[MAX_K];
            int miss_cache_idx[MAX_K];  // cache entry index for each miss
            int num_misses = 0;

            for (int k = 0; k < actual_K; k++) {
                id<MTLBuffer> cached = malloc_cache_lookup(g_malloc_cache, layer_idx, expert_indices[k]);
                if (cached) {
                    // Cache hit: zero-copy dispatch directly from cache buffer
                    expert_bufs[k] = cached;
                    valid[k] = 1;
                } else {
                    // Cache miss: insert entry (get buffer to pread into)
                    int cidx = -1;
                    id<MTLBuffer> buf = malloc_cache_insert(g_malloc_cache, layer_idx, expert_indices[k], &cidx);
                    expert_bufs[k] = buf;
                    miss_indices[num_misses] = k;
                    miss_cache_idx[num_misses] = cidx;
                    num_misses++;
                    valid[k] = 0;
                }
            }

            // Phase 2: parallel pread misses directly into cache buffers (zero-copy)
            if (num_misses > 0) {
                InferPreadTask tasks[MAX_K];
                for (int m = 0; m < num_misses; m++) {
                    int k = miss_indices[m];
                    int cidx = miss_cache_idx[m];
                    off_t eoff; size_t esz;
                    expert_offset_size(layer_idx, expert_indices[k], &eoff, &esz);
                    tasks[m].fd = expert_pick_fd(layer_idx, expert_indices[k], packed_fd);
                    tasks[m].dst = g_malloc_cache->data[cidx];
                    tasks[m].offset = eoff;
                    tasks[m].size = esz;
                    tasks[m].result = 0;
                    tasks[m].mmap_base = NULL;  // always pread for cache population
                }

                io_pool_dispatch(tasks, num_misses);

                // Mark valid
                for (int m = 0; m < num_misses; m++) {
                    int k = miss_indices[m];
                    valid[k] = (tasks[m].result == (ssize_t)tasks[m].size);
                    if (!valid[k]) {
                        fprintf(stderr, "WARNING: expert %d pread: %zd/%zu\n",
                                expert_indices[k], tasks[m].result, tasks[m].size);
                    }
                }
            }
        } else if (g_expert_cache) {
            // ---- Metal buffer LRU cache path ----
            // Phase 1: check cache for each expert, collect misses
            int miss_indices[MAX_K];       // indices into expert_indices[] for misses
            id<MTLBuffer> miss_bufs[MAX_K]; // cache buffers to pread into
            int num_misses = 0;

            for (int k = 0; k < actual_K; k++) {
                id<MTLBuffer> cached = expert_cache_lookup(g_expert_cache, layer_idx, expert_indices[k]);
                if (cached) {
                    // Cache hit: use this buffer directly for GPU dispatch
                    expert_bufs[k] = cached;
                    valid[k] = 1;
                } else {
                    // Cache miss: insert into cache (allocates or evicts), will pread below
                    id<MTLBuffer> buf = expert_cache_insert(g_expert_cache, layer_idx, expert_indices[k]);
                    if (buf) {
                        expert_bufs[k] = buf;
                        miss_indices[num_misses] = k;
                        miss_bufs[num_misses] = buf;
                        num_misses++;
                        valid[k] = 0;  // not yet loaded
                    } else {
                        expert_bufs[k] = nil;
                        valid[k] = 0;
                    }
                }
            }

            // Phase 2: parallel pread all cache misses
            if (num_misses > 0) {
                InferPreadTask tasks[MAX_K];
                for (int m = 0; m < num_misses; m++) {
                    int k = miss_indices[m];
                    off_t eoff; size_t esz;
                    expert_offset_size(layer_idx, expert_indices[k], &eoff, &esz);
                    tasks[m].fd = expert_pick_fd(layer_idx, expert_indices[k], packed_fd);
                    tasks[m].dst = [miss_bufs[m] contents];
                    tasks[m].offset = eoff;
                    tasks[m].size = esz;
                    tasks[m].result = 0;
                    tasks[m].mmap_base = mmap_base;
                }

                io_pool_dispatch(tasks, num_misses);

                // Mark successfully loaded misses as valid
                for (int m = 0; m < num_misses; m++) {
                    int k = miss_indices[m];
                    valid[k] = (tasks[m].result == (ssize_t)tasks[m].size);
                    if (!valid[k]) {
                        fprintf(stderr, "WARNING: expert %d pread: %zd/%zu\n",
                                expert_indices[k], tasks[m].result, tasks[m].size);
                    }
                }
            }
        } else if (pred_started) {
            // ---- Prediction path: predicted experts already loading into buf_B ----
            // Wait for predicted preads (they've had ~1.6ms: CMD1_wait + attn + CMD2 + routing)
            async_pread_wait();
            g_pred_layers++;

            // Match predictions against actual routing
            int miss_ei[MAX_K];       // actual expert indices for misses
            int miss_k_slots[MAX_K];  // which k-slot each miss maps to
            int miss_count = 0;
            int hit_count = 0;

            for (int k = 0; k < actual_K; k++) {
                int found = 0;
                for (int p = 0; p < PRED_COUNT(layer_idx); p++) {
                    if (expert_indices[k] == PRED_EXPERT(layer_idx, p) &&
                        g_async_pread.valid[p]) {
                        // Hit! This expert was pre-loaded into buf_B[p]
                        expert_bufs[k] = g_metal->buf_multi_expert_data_B[p];
                        valid[k] = 1;
                        found = 1;
                        hit_count++;
                        break;
                    }
                }
                if (!found) {
                    miss_ei[miss_count] = expert_indices[k];
                    miss_k_slots[miss_count] = k;
                    expert_bufs[k] = g_metal->buf_multi_expert_data[k];
                    miss_count++;
                }
            }
            g_pred_hits += hit_count;
            g_pred_misses += miss_count;

            // Parallel sync-pread misses into buf_A
            if (miss_count > 0) {
                InferPreadTask tasks[MAX_K];
                for (int m = 0; m < miss_count; m++) {
                    int k = miss_k_slots[m];
                    off_t eoff; size_t esz;
                    expert_offset_size(layer_idx, miss_ei[m], &eoff, &esz);
                    tasks[m].fd = packed_fd;
                    tasks[m].dst = [g_metal->buf_multi_expert_data[k] contents];
                    tasks[m].offset = eoff;
                    tasks[m].size = esz;
                    tasks[m].result = 0;
                }
                io_pool_dispatch(tasks, miss_count);
                for (int m = 0; m < miss_count; m++) {
                    int k = miss_k_slots[m];
                    valid[k] = (tasks[m].result == (ssize_t)tasks[m].size);
                }
            }
        } else if (g_use_lz4 && g_lz4_index[layer_idx]) {
            // ---- LZ4 compressed path: read compressed + decompress via io_pool ----
            // Note: LZ4 + tiered is not supported (LZ4 path uses its own offsets)
            size_t esz = active_expert_size();
            InferPreadTask tasks[MAX_K];
            for (int k = 0; k < actual_K; k++) {
                LZ4IndexEntry *ie = &g_lz4_index[layer_idx][expert_indices[k]];
                tasks[k].fd = packed_fd;
                tasks[k].dst = [g_metal->buf_multi_expert_data[k] contents];
                tasks[k].offset = ie->offset;
                tasks[k].size = esz;
                tasks[k].result = 0;
                tasks[k].mmap_base = NULL;
                tasks[k].lz4_comp_buf = g_lz4_comp_bufs[k];
                tasks[k].lz4_comp_size = ie->comp_size;
                expert_bufs[k] = g_metal->buf_multi_expert_data[k];
            }
            io_pool_dispatch(tasks, actual_K);
            for (int k = 0; k < actual_K; k++) {
                valid[k] = (tasks[k].result == (ssize_t)esz);
            }
        } else {
            // ---- No cache, no prediction, no LZ4: ASYNC parallel pread ----
            async_pread_start(packed_fd, expert_indices, actual_K,
                              g_metal->buf_multi_expert_data, mmap_base,
                              layer_idx);
            for (int k = 0; k < actual_K; k++) {
                expert_bufs[k] = g_metal->buf_multi_expert_data[k];
            }
        }

        // Shared expert prep (doesn't need expert data — can overlap with async pread)
        memcpy([g_metal->buf_multi_expert_input contents], h_post, cfg.hidden_dim * sizeof(float));
        memcpy([g_metal->buf_shared_gate contents], shared_gate,
               cfg.shared_intermediate * sizeof(float));
        memcpy([g_metal->buf_shared_up contents], shared_up,
               cfg.shared_intermediate * sizeof(float));

        // Wait for non-prediction async pread to complete
        if (!pred_started && g_async_pread.active) {
            async_pread_wait();
            for (int k = 0; k < actual_K; k++) {
                valid[k] = g_async_pread.valid[k];
            }
        }

        if (g_timing_enabled) { t1 = now_ms(); g_timing.expert_io += t1 - t0; }

        // Store this layer's routing for next token's temporal prediction.
        // MUST happen AFTER the prediction hit check above (which reads g_pred_experts).
        if (g_pred_enabled && g_pred_generating) {
            for (int k = 0; k < actual_K; k++) {
                PRED_EXPERT(layer_idx, k) = expert_indices[k];
            }
            PRED_COUNT(layer_idx) = actual_K;
            if (layer_idx == cfg.num_layers - 1) {
                g_pred_valid = 1;
            }
        }

        if (g_timing_enabled) { t0 = now_ms(); }

        // Step 3: encode ALL experts + shared expert into ONE command buffer.
        // Batched encoding: 4 encoders for K experts + 2 for shared = 6 total
        // (vs. 4*K + 2 = 18 with old per-expert encoding).
        id<MTLCommandBuffer> cmd_experts = [g_metal->queue commandBuffer];

        gpu_encode_experts_batched(g_metal, cmd_experts, actual_K, valid, expert_bufs,
                                   layer_idx, expert_indices);

        // Shared expert SwiGLU + down_proj (2 more encoders)
        // Note: shared_gate/up already copied to GPU buffers above (before async pread wait)

        // SwiGLU dispatch
        {
            id<MTLComputeCommandEncoder> enc = [cmd_experts computeCommandEncoder];
            [enc setComputePipelineState:g_metal->swiglu];
            [enc setBuffer:g_metal->buf_shared_gate offset:0 atIndex:0];
            [enc setBuffer:g_metal->buf_shared_up   offset:0 atIndex:1];
            [enc setBuffer:g_metal->buf_shared_act  offset:0 atIndex:2];
            uint32_t dim = cfg.shared_intermediate;
            [enc setBytes:&dim length:4 atIndex:3];
            uint32_t swiglu_tgs = (dim + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // Shared down_proj dispatch
        if (sdw && sds && sdb) {
            if (g_metal->wf_num_chunks > 0) {
                // GPU path: weight data accessible via Metal buffer
                gpu_encode_dequant_matvec_with_io_bufs(
                    g_metal, cmd_experts, sdw, sds, sdb,
                    g_metal->buf_shared_act, g_metal->buf_shared_out,
                    cfg.hidden_dim, cfg.shared_intermediate, cfg.group_size);
            } else {
                // CPU fallback: weight file too large for Metal buffers
                float *shared_act_cpu = (float *)[g_metal->buf_shared_act contents];
                float *shared_out_cpu = (float *)[g_metal->buf_shared_out contents];
                cpu_dequant_matvec((const uint32_t *)sdw, (const uint16_t *)sds,
                                   (const uint16_t *)sdb, shared_act_cpu, shared_out_cpu,
                                   cfg.hidden_dim, cfg.shared_intermediate, cfg.group_size);
            }
        }

        // Step 4: GPU-side combine + residual + norm (if not last layer)
        // Appends dispatches to CMD3 so the next layer's CMD1 can submit immediately
        // without waiting for CMD3 to complete + CPU readback.
        //
        // For non-last layers with the combine pipeline available:
        //   Enc C1: moe_combine_residual (expert_outs + h_mid + shared_out -> buf_moe_hidden)
        //   Enc C2: rms_norm_sum_sq (buf_moe_hidden -> buf_cmd3_sum_sq)
        //   Enc C3: rms_norm_apply_bf16 (buf_moe_hidden + next_layer_norm_w -> buf_input)
        //
        // This makes CMD3 self-contained: it produces buf_input for the next layer's CMD1.
        // The next layer skips deferred_wait + finalize + input_norm entirely at layer start.

        int gpu_combine = (g_metal->moe_combine_residual &&
                           g_metal->rms_norm_sum &&
                           g_metal->rms_norm_apply_bf16 &&
                           g_metal->wf_buf &&
                           layer_idx < cfg.num_layers - 1 &&
                           layer_cache[layer_idx + 1].input_norm_w != NULL);

        if (gpu_combine) {
            // Copy h_mid from buf_h_mid (populated by CMD2) — it's still valid on GPU.
            // h_mid is already in buf_h_mid from CMD2's residual_add dispatch.

            // Prepare combine params: expert_weights[0..K-1] + shared_gate_score
            {
                float *params = (float *)[g_metal->buf_combine_params contents];
                // Zero all 10 slots first (unused experts get weight=0)
                memset(params, 0, 10 * sizeof(float));
                for (int k = 0; k < actual_K; k++) {
                    params[k] = valid[k] ? expert_weights[k] : 0.0f;
                }
                params[8] = shared_gate_score;
            }

            // Enc C1: moe_combine_residual
            {
                id<MTLComputeCommandEncoder> enc = [cmd_experts computeCommandEncoder];
                [enc setComputePipelineState:g_metal->moe_combine_residual];
                [enc setBuffer:g_metal->buf_h_mid         offset:0 atIndex:0];   // h_mid
                [enc setBuffer:g_metal->buf_shared_out    offset:0 atIndex:1];   // shared_out
                [enc setBuffer:g_metal->buf_moe_hidden    offset:0 atIndex:2];   // output: hidden
                // Bind all 8 expert output buffers (unused ones have weight=0 in params)
                for (int k = 0; k < MAX_K; k++) {
                    [enc setBuffer:g_metal->buf_multi_expert_out[k] offset:0 atIndex:(3 + k)];
                }
                [enc setBuffer:g_metal->buf_combine_params offset:0 atIndex:11]; // params
                uint32_t dim = cfg.hidden_dim;
                uint32_t k_val = (uint32_t)actual_K;
                [enc setBytes:&dim   length:4 atIndex:12];
                [enc setBytes:&k_val length:4 atIndex:13];
                uint32_t tgs = (dim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }

            // Enc C2: rms_norm_sum_sq (buf_moe_hidden -> buf_cmd3_sum_sq)
            {
                id<MTLComputeCommandEncoder> enc = [cmd_experts computeCommandEncoder];
                uint32_t dim = cfg.hidden_dim;
                [enc setComputePipelineState:g_metal->rms_norm_sum];
                [enc setBuffer:g_metal->buf_moe_hidden  offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_cmd3_sum_sq offset:0 atIndex:1];
                [enc setBytes:&dim length:4 atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }

            // Enc C3: rms_norm_apply_bf16 (buf_moe_hidden + next_norm_w -> buf_input)
            {
                uint16_t *next_norm_w = layer_cache[layer_idx + 1].input_norm_w;
                id<MTLBuffer> nnw_buf; NSUInteger nnw_off;
                metal_staging_reset(g_metal);
                metal_find_chunk_sized(g_metal, next_norm_w,
                    (size_t)cfg.hidden_dim * 2, &nnw_buf, &nnw_off);  // bf16
                id<MTLComputeCommandEncoder> enc = [cmd_experts computeCommandEncoder];
                uint32_t dim = cfg.hidden_dim;
                float eps = cfg.rms_norm_eps;
                [enc setComputePipelineState:g_metal->rms_norm_apply_bf16];
                [enc setBuffer:g_metal->buf_moe_hidden  offset:0       atIndex:0]; // x
                [enc setBuffer:nnw_buf offset:nnw_off atIndex:1]; // weight (bf16)
                [enc setBuffer:g_metal->buf_cmd3_sum_sq offset:0       atIndex:2]; // sum_sq
                [enc setBuffer:g_metal->buf_input       offset:0       atIndex:3]; // out = normed
                [enc setBytes:&dim length:4 atIndex:4];
                [enc setBytes:&eps length:4 atIndex:5];
                uint32_t tgs = (dim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
        }

        // DEFERRED commit — submit async, don't wait.
        [cmd_experts commit];
        if (g_timing_enabled) {
            t1 = now_ms();
            g_timing.cmd3_encode += t1 - t0;
            g_timing.count++;
            g_timing.total += t1 - t_layer_start;
        }

        // Save state for deferred completion
        g_deferred.active = 1;
        g_deferred.gpu_combined = gpu_combine;
        g_deferred.cmd_experts = cmd_experts;
        g_deferred.actual_K = actual_K;
        g_deferred.shared_gate_score = shared_gate_score;
        g_deferred.hidden = hidden;
        g_deferred.layer_idx = layer_idx;
        if (!gpu_combine) {
            // Only need to save h_mid for CPU-side combine path
            memcpy(g_deferred.h_mid, h_mid, cfg.hidden_dim * sizeof(float));
        }
        for (int k = 0; k < actual_K; k++) {
            g_deferred.expert_weights[k] = expert_weights[k];
            g_deferred.valid[k] = valid[k];
        }

        // Expert cross-layer prefetch: start pread'ing next layer's predicted experts
        // into buffer set B. The I/O overlaps with GPU compute (CMD3 is running async).
        // Uses temporal prediction (last token's expert choices for layer+1).
        // NOTE: Each layer has its own packed_fd. Cross-layer prefetch requires the
        // caller to store layer fds globally. When g_expert_prefetch_layer_fds is set
        // (by InferContext setup), this reads the next layer's predicted experts.
        if (g_expert_prefetch_enabled && g_pred_enabled && g_pred_generating &&
            g_pred_valid && layer_idx + 1 < cfg.num_layers &&
            g_metal->buf_multi_expert_data_B[0] &&
            g_expert_prefetch_layer_fds &&
            PRED_COUNT(layer_idx + 1) > 0) {
            int next_fd = g_expert_prefetch_layer_fds[layer_idx + 1];
            if (next_fd >= 0) {
                async_pread_start(next_fd, &PRED_EXPERT(layer_idx + 1, 0),
                                  PRED_COUNT(layer_idx + 1),
                                  g_metal->buf_multi_expert_data_B, mmap_base,
                                  layer_idx + 1);
                g_prefetch_hits_total++;
            }
        }

        // Return immediately — GPU experts are running async.
        // The next call to fused_layer_forward() or complete_deferred_experts()
        // will wait for the GPU and apply the final combine.
        g_use_2bit = saved_use_2bit;
        g_use_q3_outlier = saved_use_q3_outlier;
        g_use_q3_experts = saved_use_q3_experts;
        g_active_q3_layout = saved_active_q3_layout;
        g_active_q3_layout_valid = saved_active_q3_layout_valid;
        return;

    } else if (packed_fd >= 0) {
        // CPU fallback for experts
        float *expert_out_cpu = malloc(cfg.hidden_dim * sizeof(float));
        for (int k = 0; k < K; k++) {
            int eidx = expert_indices[k];
            off_t expert_offset; size_t esz;
            expert_offset_size(layer_idx, eidx, &expert_offset, &esz);
            void *expert_data = malloc(esz);
            ssize_t nread = pread(packed_fd, expert_data, esz, expert_offset);
            if (nread != (ssize_t)esz) {
                fprintf(stderr, "WARNING: layer %d expert %d pread: %zd/%zu\n",
                        layer_idx, eidx, nread, esz);
                free(expert_data);
                continue;
            }

            // CPU fallback offsets — determine quant per expert
            int use_2bit_k = g_use_2bit;
            if (g_use_tiered && g_tiered_manifest)
                use_2bit_k = (TIERED(layer_idx, eidx).bits == 2);
            uint32_t *gw = (uint32_t *)expert_data;
            uint16_t *gs_p = (uint16_t *)((char *)expert_data + (use_2bit_k ? cfg.gate_s_off_2 : cfg.gate_s_off_4));
            uint16_t *gb_p = (uint16_t *)((char *)expert_data + (use_2bit_k ? cfg.gate_b_off_2 : cfg.gate_b_off_4));
            uint32_t *uw = (uint32_t *)((char *)expert_data + (use_2bit_k ? cfg.up_w_off_2 : cfg.up_w_off_4));
            uint16_t *us_p = (uint16_t *)((char *)expert_data + (use_2bit_k ? cfg.up_s_off_2 : cfg.up_s_off_4));
            uint16_t *ub_p = (uint16_t *)((char *)expert_data + (use_2bit_k ? cfg.up_b_off_2 : cfg.up_b_off_4));
            uint32_t *dw = (uint32_t *)((char *)expert_data + (use_2bit_k ? cfg.down_w_off_2 : cfg.down_w_off_4));
            uint16_t *ds_p = (uint16_t *)((char *)expert_data + (use_2bit_k ? cfg.down_s_off_2 : cfg.down_s_off_4));
            uint16_t *db_p = (uint16_t *)((char *)expert_data + (use_2bit_k ? cfg.down_b_off_2 : cfg.down_b_off_4));

            float *gate_proj_out = malloc(cfg.moe_intermediate * sizeof(float));
            float *up_proj_out = malloc(cfg.moe_intermediate * sizeof(float));
            float *act_out = malloc(cfg.moe_intermediate * sizeof(float));

            cpu_dequant_matvec(gw, gs_p, gb_p, h_post, gate_proj_out,
                               cfg.moe_intermediate, cfg.hidden_dim, cfg.group_size);
            cpu_dequant_matvec(uw, us_p, ub_p, h_post, up_proj_out,
                               cfg.moe_intermediate, cfg.hidden_dim, cfg.group_size);
            cpu_swiglu(gate_proj_out, up_proj_out, act_out, cfg.moe_intermediate);
            cpu_dequant_matvec(dw, ds_p, db_p, act_out, expert_out_cpu,
                               cfg.hidden_dim, cfg.moe_intermediate, cfg.group_size);

            free(gate_proj_out);
            free(up_proj_out);
            free(act_out);
            free(expert_data);

            cpu_vec_madd(moe_out, expert_out_cpu, expert_weights[k], cfg.hidden_dim);
        }
        free(expert_out_cpu);

        // CPU shared expert
        float *shared_act = calloc(cfg.shared_intermediate, sizeof(float));
        cpu_swiglu(shared_gate, shared_up, shared_act, cfg.shared_intermediate);
        if (sdw && sds && sdb) {
            cpu_dequant_matvec(sdw, sds, sdb, shared_act, shared_out,
                               cfg.hidden_dim, cfg.shared_intermediate, cfg.group_size);
        }
        free(shared_act);
    } else {
        // No experts available -- still need shared expert
        float *shared_act = calloc(cfg.shared_intermediate, sizeof(float));
        cpu_swiglu(shared_gate, shared_up, shared_act, cfg.shared_intermediate);
        if (sdw && sds && sdb) {
            fast_dequant_matvec(sdw, sds, sdb, shared_act, shared_out,
                                cfg.hidden_dim, cfg.shared_intermediate, cfg.group_size);
        }
        free(shared_act);
    }

    // ---- Shared expert gate ----
    float shared_weight = cpu_sigmoid(shared_gate_score);
    for (int i = 0; i < cfg.hidden_dim; i++) {
        shared_out[i] *= shared_weight;
    }

    // ---- Final combine: hidden = h_mid + moe_out + shared_out ----
    for (int i = 0; i < cfg.hidden_dim; i++) {
        hidden[i] = h_mid[i] + moe_out[i] + shared_out[i];
    }

    if (g_timing_enabled) {
        t1 = now_ms();
        g_timing.cmd3_encode += t1 - t0;  // includes CPU expert compute for non-GPU paths
        g_timing.count++;
        g_timing.total += t1 - t_layer_start;
    }

    // h_post, h_mid, gate_scores, moe_out, shared_out, shared_gate, shared_up
    // are all static scratch buffers — no free needed.

    // Restore global expert quant flag(s)
    g_use_2bit = saved_use_2bit;
    g_use_q3_outlier = saved_use_q3_outlier;
    g_use_q3_experts = saved_use_q3_experts;
    g_active_q3_layout = saved_active_q3_layout;
    g_active_q3_layout_valid = saved_active_q3_layout_valid;
}

// ============================================================================
// Main inference loop
// ============================================================================

// ============================================================================
// Expert frequency analysis (--freq)
// ============================================================================

static int freq_cmp_desc(const void *a, const void *b) {
    return *(const int *)b - *(const int *)a;
}

static void freq_print_analysis(int K) {
    if (!g_freq_tracking || g_freq_total_tokens == 0) return;

    int total_activations_per_layer = g_freq_total_tokens * K;

    fprintf(stderr, "\n=== Expert Frequency Analysis ===\n");
    fprintf(stderr, "Tokens tracked: %d, K=%d, activations/layer=%d\n\n",
            g_freq_total_tokens, K, total_activations_per_layer);

    // Per-layer analysis
    int experts_for_80_total = 0;  // sum across layers for overall estimate

    for (int l = 0; l < cfg.num_layers; l++) {
        // Count unique experts and sort frequencies descending
        int sorted[cfg.num_experts];
        memcpy(sorted, &FREQ(l, 0), cfg.num_experts * sizeof(int));
        qsort(sorted, cfg.num_experts, sizeof(int), freq_cmp_desc);

        int unique = 0;
        for (int e = 0; e < cfg.num_experts; e++) {
            if (sorted[e] > 0) unique++;
        }

        // Compute cumulative coverage thresholds
        int cum = 0;
        int top10_cov = 0, top30_cov = 0, top60_cov = 0;
        int n_for_50 = 0, n_for_80 = 0, n_for_90 = 0;
        for (int e = 0; e < cfg.num_experts; e++) {
            cum += sorted[e];
            if (e == 9)  top10_cov = cum;
            if (e == 29) top30_cov = cum;
            if (e == 59) top60_cov = cum;
            if (n_for_50 == 0 && cum * 100 >= total_activations_per_layer * 50)
                n_for_50 = e + 1;
            if (n_for_80 == 0 && cum * 100 >= total_activations_per_layer * 80)
                n_for_80 = e + 1;
            if (n_for_90 == 0 && cum * 100 >= total_activations_per_layer * 90)
                n_for_90 = e + 1;
        }

        double pct10 = 100.0 * top10_cov / total_activations_per_layer;
        double pct30 = 100.0 * top30_cov / total_activations_per_layer;
        double pct60 = 100.0 * top60_cov / total_activations_per_layer;

        fprintf(stderr, "Layer %2d: %3d unique experts, "
                "top-10 cover %.0f%%, top-30 cover %.0f%%, top-60 cover %.0f%% "
                "(50%%@%d, 80%%@%d, 90%%@%d)\n",
                l, unique, pct10, pct30, pct60, n_for_50, n_for_80, n_for_90);

        experts_for_80_total += n_for_80;
    }

    // Overall summary: average experts needed for 80% across all layers
    double avg_experts_80 = (double)experts_for_80_total / cfg.num_layers;
    // Expert size in GB: each expert is active_expert_size() bytes
    double expert_gb = (double)active_expert_size() / (1024.0 * 1024.0 * 1024.0);
    double total_pin_gb = avg_experts_80 * cfg.num_layers * expert_gb;

    fprintf(stderr, "\n--- Overall Summary ---\n");
    fprintf(stderr, "To achieve 80%% hit rate across all layers, need %d experts pinned "
            "(avg %.0f/layer, %.2f GB)\n",
            experts_for_80_total, avg_experts_80, total_pin_gb);
    fprintf(stderr, "Expert size: %zu bytes (%.3f MB), %d layers x %d experts = %d total\n",
            active_expert_size(), (double)active_expert_size() / (1024.0 * 1024.0),
            cfg.num_layers, cfg.num_experts, cfg.num_layers * cfg.num_experts);

    // Raw frequency dump for profile_experts.py
    fprintf(stderr, "\n--- Raw Frequency Dump (for profile_experts.py) ---\n");
    for (int l = 0; l < cfg.num_layers; l++) {
        fprintf(stderr, "FREQ_DUMP layer=%d:", l);
        for (int e = 0; e < cfg.num_experts; e++) {
            int f = FREQ(l, e);
            if (f > 0) fprintf(stderr, " %d:%d", e, f);
        }
        fprintf(stderr, "\n");
    }
}

// Tokenize a continuation turn (available in both CLI and iOS modes).
// Prefixes with \n<|im_start|>user\n to start new turn, assumes prior assistant
// turn's EOS/<|im_end|> is already in the KV cache state.
static PromptTokens *tokenize_continuation_turn_shared(const char *user_content) {
    const char *prefix = "\n<|im_start|>user\n";
    const char *suffix = "<|im_end|>\n<|im_start|>assistant\n";

    size_t prompt_len = strlen(prefix) + strlen(user_content) + strlen(suffix) + 1;
    char *prompt = malloc(prompt_len);
    if (!prompt) return NULL;
    snprintf(prompt, prompt_len, "%s%s%s", prefix, user_content, suffix);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}

#ifndef CHAT_MODE

// ============================================================================
// HTTP Serve Mode — OpenAI-compatible /v1/chat/completions (SSE streaming)
// ============================================================================

// Read exactly n bytes from fd, returns 0 on success, -1 on error/EOF
static int read_exact(int fd, char *buf, int n) {
    int got = 0;
    while (got < n) {
        ssize_t r = read(fd, buf + got, n - got);
        if (r <= 0) return -1;
        got += (int)r;
    }
    return 0;
}

// Read HTTP request into buf (up to bufsz-1). Returns total bytes read, or -1.
// Reads headers, then Content-Length body if present.
static int read_http_request(int fd, char *buf, int bufsz) {
    int total = 0;
    // Read until we find \r\n\r\n (end of headers)
    while (total < bufsz - 1) {
        ssize_t r = read(fd, buf + total, 1);
        if (r <= 0) return -1;
        total++;
        if (total >= 4 &&
            buf[total-4] == '\r' && buf[total-3] == '\n' &&
            buf[total-2] == '\r' && buf[total-1] == '\n') {
            break;
        }
    }
    buf[total] = '\0';

    // Find Content-Length
    const char *cl = strcasestr(buf, "Content-Length:");
    if (cl) {
        int content_len = atoi(cl + 15);
        if (content_len > 0 && total + content_len < bufsz - 1) {
            if (read_exact(fd, buf + total, content_len) < 0) return -1;
            total += content_len;
            buf[total] = '\0';
        }
    }
    return total;
}

// Extract the last "content" value from an OpenAI messages array.
// Minimal JSON parsing: find last "content":" and extract the string value.
// Returns pointer into buf (null-terminated in place), or NULL.
static char *extract_last_content(char *buf) {
    char *last = NULL;
    char *p = buf;
    for (;;) {
        p = strstr(p, "\"content\"");
        if (!p) break;
        p += 9; // skip "content"
        // Skip whitespace and colon
        while (*p == ' ' || *p == '\t' || *p == ':') p++;
        if (*p == '"') {
            p++; // skip opening quote
            last = p;
            // Find closing quote (handle escapes)
            while (*p && !(*p == '"' && *(p-1) != '\\')) p++;
        }
    }
    if (last) {
        // Null-terminate the content string (overwrite closing quote)
        char *end = last;
        while (*end && !(*end == '"' && (end == last || *(end-1) != '\\'))) end++;
        *end = '\0';
        // Unescape \\n -> \n, \\" -> ", \\\\ -> backslash inline
        char *r = last, *w = last;
        while (*r) {
            if (*r == '\\' && *(r+1)) {
                r++;
                switch (*r) {
                    case 'n':  *w++ = '\n'; r++; break;
                    case 't':  *w++ = '\t'; r++; break;
                    case '"':  *w++ = '"';  r++; break;
                    case '\\': *w++ = '\\'; r++; break;
                    default:   *w++ = '\\'; *w++ = *r++; break;
                }
            } else {
                *w++ = *r++;
            }
        }
        *w = '\0';
    }
    return last;
}

// Extract "max_tokens" or "max_completion_tokens" from JSON body. Returns value or default.
static int extract_max_tokens(const char *buf, int default_val) {
    const char *p = strstr(buf, "\"max_completion_tokens\"");
    if (!p) p = strstr(buf, "\"max_tokens\"");
    if (!p) return default_val;
    p = strchr(p, ':');
    if (!p) return default_val;
    return atoi(p + 1);
}

// Save a conversation turn to ~/.flash-moe/sessions/<session_id>.jsonl
// Shared data store with the chat client.
static void server_save_turn(const char *session_id, const char *role, const char *content) {
    if (!session_id || !session_id[0] || !content) return;
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    char dir[1024], path[1024];
    snprintf(dir, sizeof(dir), "%s/.flash-moe/sessions", home);
    mkdir(dir, 0755);
    char parent[1024];
    snprintf(parent, sizeof(parent), "%s/.flash-moe", home);
    mkdir(parent, 0755);
    mkdir(dir, 0755);
    snprintf(path, sizeof(path), "%s/%s.jsonl", dir, session_id);
    FILE *f = fopen(path, "a");
    if (!f) return;
    // JSON-escape content
    size_t clen = strlen(content);
    char *escaped = malloc(clen * 2 + 1);
    int j = 0;
    for (size_t i = 0; i < clen; i++) {
        switch (content[i]) {
            case '"': escaped[j++]='\\'; escaped[j++]='"'; break;
            case '\\': escaped[j++]='\\'; escaped[j++]='\\'; break;
            case '\n': escaped[j++]='\\'; escaped[j++]='n'; break;
            case '\r': escaped[j++]='\\'; escaped[j++]='r'; break;
            case '\t': escaped[j++]='\\'; escaped[j++]='t'; break;
            default: escaped[j++]=content[i]; break;
        }
    }
    escaped[j] = 0;
    fprintf(f, "{\"role\":\"%s\",\"content\":\"%s\"}\n", role, escaped);
    free(escaped);
    fclose(f);
}

// Extract "session_id" string from JSON body. Copies into out_buf (max out_size).
// Returns 1 if found, 0 if missing.
static int extract_session_id(const char *buf, char *out_buf, int out_size) {
    const char *p = strstr(buf, "\"session_id\"");
    if (!p) return 0;
    p += 12; // skip "session_id"
    while (*p == ' ' || *p == '\t' || *p == ':') p++;
    if (*p != '"') return 0;
    p++; // skip opening quote
    int i = 0;
    while (*p && *p != '"' && i < out_size - 1) {
        out_buf[i++] = *p++;
    }
    out_buf[i] = '\0';
    return i > 0 ? 1 : 0;
}

// Write a full HTTP response string to fd
static void http_write(int fd, const char *data, int len) {
    int sent = 0;
    while (sent < len) {
        ssize_t w = write(fd, data + sent, len - sent);
        if (w <= 0) break;
        sent += (int)w;
    }
}

static void http_write_str(int fd, const char *s) {
    http_write(fd, s, (int)strlen(s));
}

// ============================================================================
// UTF-8 streaming buffer for serve_loop
// ============================================================================
// BPE tokens can split multi-byte UTF-8 sequences across token boundaries
// (e.g., emoji 👋 = F0 9F 91 8B may be two tokens: [F0 9F] [91 8B]).
// Each fragment alone is invalid UTF-8, so we buffer partial sequences and
// only emit complete UTF-8 codepoints.

typedef struct {
    char  pending[8];   // at most 3 trailing bytes of incomplete sequence
    int   pending_len;
} ServeUtf8Buf;

// Returns number of bytes from the END of buf that form an incomplete UTF-8 sequence.
// 0 means the entire buffer is valid UTF-8.
static int serve_utf8_incomplete_tail(const char *buf, int len) {
    if (len == 0) return 0;
    for (int i = 1; i <= 4 && i <= len; i++) {
        unsigned char c = (unsigned char)buf[len - i];
        if ((c & 0x80) == 0) return 0;  // ASCII — complete
        if ((c & 0xC0) == 0xC0) {
            int expected;
            if ((c & 0xE0) == 0xC0) expected = 2;
            else if ((c & 0xF0) == 0xE0) expected = 3;
            else if ((c & 0xF8) == 0xF0) expected = 4;
            else return i;  // invalid start byte
            if (i >= expected) return 0;  // sequence is complete
            return i;  // incomplete
        }
    }
    return len < 4 ? len : 4;
}

// Push token bytes through the UTF-8 buffer. Returns pointer to complete UTF-8 string
// to emit (may be empty ""). out buffer must be at least 256 bytes.
static const char *serve_utf8_push(ServeUtf8Buf *u, const char *data, int len,
                                    char *out, int out_size) {
    int total = u->pending_len + len;
    if (total >= out_size - 1) total = out_size - 2;

    if (u->pending_len > 0) memcpy(out, u->pending, u->pending_len);
    int copy_len = total - u->pending_len;
    if (copy_len > 0) memcpy(out + u->pending_len, data, copy_len);
    out[total] = '\0';

    int tail = serve_utf8_incomplete_tail(out, total);
    if (tail > 0) {
        memcpy(u->pending, out + total - tail, tail);
        u->pending_len = tail;
        out[total - tail] = '\0';
    } else {
        u->pending_len = 0;
    }
    return out;
}

// Flush remaining pending bytes (at end of generation)
static const char *serve_utf8_flush(ServeUtf8Buf *u, char *out, int out_size) {
    if (u->pending_len > 0 && u->pending_len < out_size - 1) {
        memcpy(out, u->pending, u->pending_len);
        out[u->pending_len] = '\0';
        u->pending_len = 0;
        return out;
    }
    out[0] = '\0';
    return out;
}

// JSON-escape a string for SSE delta content, handling all control chars
static int serve_json_escape(const char *src, char *dst, int dst_size) {
    int j = 0;
    for (int i = 0; src[i] && j < dst_size - 6; i++) {
        unsigned char c = (unsigned char)src[i];
        switch (c) {
            case '"':  dst[j++] = '\\'; dst[j++] = '"'; break;
            case '\\': dst[j++] = '\\'; dst[j++] = '\\'; break;
            case '\n': dst[j++] = '\\'; dst[j++] = 'n'; break;
            case '\r': dst[j++] = '\\'; dst[j++] = 'r'; break;
            case '\t': dst[j++] = '\\'; dst[j++] = 't'; break;
            case '\b': dst[j++] = '\\'; dst[j++] = 'b'; break;
            case '\f': dst[j++] = '\\'; dst[j++] = 'f'; break;
            default:
                if (c < 0x20) {
                    j += snprintf(dst + j, dst_size - j, "\\u%04x", c);
                } else {
                    dst[j++] = c;
                }
                break;
        }
    }
    dst[j] = '\0';
    return j;
}

// Send an SSE chunk with a token delta (UTF-8 safe content)
// Returns 0 on success, -1 if client disconnected
static int sse_send_delta(int fd, const char *request_id, const char *token_text) {
    if (!token_text || !token_text[0]) return 0;

    char chunk[4096];
    char escaped[2048];
    serve_json_escape(token_text, escaped, sizeof(escaped));

    int n = snprintf(chunk, sizeof(chunk),
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
        request_id, escaped);
    if (n >= (int)sizeof(chunk)) n = (int)sizeof(chunk) - 1; // clamp to buffer size
    ssize_t wr = write(fd, chunk, n);
    return (wr <= 0) ? -1 : 0;
}

static void sse_send_done(int fd, const char *request_id) {
    char chunk[1024];
    int n = snprintf(chunk, sizeof(chunk),
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
        "data: [DONE]\n\n",
        request_id);
    http_write(fd, chunk, n);
}

static const char *SSE_HEADERS =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: text/event-stream\r\n"
    "Cache-Control: no-cache\r\n"
    "Connection: close\r\n"
    "Access-Control-Allow-Origin: *\r\n"
    "\r\n";

static const char *CORS_RESPONSE =
    "HTTP/1.1 204 No Content\r\n"
    "Access-Control-Allow-Origin: *\r\n"
    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
    "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
    "Access-Control-Max-Age: 86400\r\n"
    "\r\n";

// Tokenize a user turn (system prompt already cached in KV).
// Only encodes: <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n
PromptTokens *tokenize_user_turn(const char *user_content) {
    const char *prefix = "<|im_start|>user\n";
    const char *suffix = "<|im_end|>\n<|im_start|>assistant\n";

    size_t prompt_len = strlen(prefix) + strlen(user_content) + strlen(suffix) + 1;
    char *prompt = malloc(prompt_len);
    if (!prompt) return NULL;
    snprintf(prompt, prompt_len, "%s%s%s", prefix, user_content, suffix);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}

// Tokenize a continuation turn for session caching.
// Prefixes with <|im_end|>\n to close the previous assistant turn, then the new user turn.
// Used when the KV cache already contains the prior conversation state.
PromptTokens *tokenize_continuation_turn(const char *user_content) {
    // EOS/<|im_end|> is already in the state (fed through model at end of generation)
    // Just need the newline + new user turn + assistant prompt
    const char *prefix = "\n<|im_start|>user\n";
    const char *suffix = "<|im_end|>\n<|im_start|>assistant\n";

    size_t prompt_len = strlen(prefix) + strlen(user_content) + strlen(suffix) + 1;
    char *prompt = malloc(prompt_len);
    if (!prompt) return NULL;
    snprintf(prompt, prompt_len, "%s%s%s", prefix, user_content, suffix);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}

// Load custom system prompt from ~/.flash-moe/system.md, or use default
char *load_system_prompt(void) {
    const char *home = getenv("HOME");
    if (home) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/.flash-moe/system.md", home);
        FILE *f = fopen(path, "r");
        if (f) {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            char *buf = malloc(sz + 1);
            size_t n = fread(buf, 1, sz, f);
            buf[n] = 0;
            fclose(f);
            fprintf(stderr, "[serve] Loaded custom system prompt from %s (%ld bytes)\n", path, sz);
            return buf;
        }
    }
    return strdup("You are a helpful assistant. /think");
}

// Tokenize a full chat message (system prompt + user turn) for first-time use.
PromptTokens *tokenize_chat_message(const char *user_content) {
    static char *sys_prompt_text = NULL;
    if (!sys_prompt_text) sys_prompt_text = load_system_prompt();

    // Build: <|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n
    size_t sys_len = strlen(sys_prompt_text);
    size_t user_len = strlen(user_content);
    size_t total = 30 + sys_len + 30 + user_len + 40;  // generous padding for tags
    char *prompt = malloc(total);
    if (!prompt) return NULL;
    snprintf(prompt, total, "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
             sys_prompt_text, user_content);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}

// Keep old signature for backward compat (unused but prevents compiler warning)
__attribute__((unused))
static PromptTokens *tokenize_chat_message_old(const char *user_content) {
    const char *prefix =
        "<|im_start|>system\nYou are a helpful assistant. /think<|im_end|>\n"
        "<|im_start|>user\n";
    const char *suffix = "<|im_end|>\n<|im_start|>assistant\n";

    size_t prompt_len = strlen(prefix) + strlen(user_content) + strlen(suffix) + 1;
    char *prompt = malloc(prompt_len);
    if (!prompt) return NULL;

    snprintf(prompt, prompt_len, "%s%s%s", prefix, user_content, suffix);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}

// The main serve loop. Model state must already be initialized.
// Sync CPU linear attention state → GPU buffers
void sync_cpu_to_gpu_delta_state_serve(void **layer_states) {
    if (!g_metal || !g_metal->delta_net_step || !layer_states) return;
    int li = 0;
    for (int i = 0; i < cfg.num_layers; i++) {
        if (cfg.is_full_attn[i]) continue;
        if (!layer_states[i]) { li++; continue; }
        LinearAttnState *la = (LinearAttnState *)layer_states[i];
        if (li < cfg.num_linear_layers) {
            if (g_metal->buf_delta_state[li] && la->ssm_state)
                memcpy([g_metal->buf_delta_state[li] contents], la->ssm_state,
                       cfg.linear_num_v_heads * cfg.linear_value_dim * cfg.linear_key_dim * sizeof(float));
            if (g_metal->buf_conv_state[li] && la->conv_state)
                memcpy([g_metal->buf_conv_state[li] contents], la->conv_state,
                       (cfg.conv_kernel_size - 1) * cfg.linear_conv_dim * sizeof(float));
        }
        li++;
    }
}

static void serve_loop(
    int port,
    WeightFile *wf, Vocabulary *vocab,
    void **layer_states, KVCache **kv_caches,
    void **layer_mmaps, int *layer_fds,
    float *hidden, float *logits,
    uint16_t *final_norm_w, int K)
{
    // Ignore SIGPIPE (client disconnect mid-write)
    signal(SIGPIPE, SIG_IGN);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return; }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(server_fd); return;
    }
    if (listen(server_fd, 8) < 0) {
        perror("listen"); close(server_fd); return;
    }

    printf("[serve] Listening on http://0.0.0.0:%d\n", port);
    printf("[serve] Endpoints: POST /v1/chat/completions, GET /v1/models, GET /health\n");
    fflush(stdout);

    static uint64_t req_counter = 0;

    // ---- System prompt cache: prefill system prompt once at startup ----
    // Tokenize the system prompt and run it through all 40 layers.
    // Save the resulting KV cache + linear attention state as a snapshot.
    // On each request, restore the snapshot instead of re-prefilling.
    fprintf(stderr, "[serve] Pre-caching system prompt...\n");
    PromptTokens *sys_pt = tokenize_chat_message("");  // empty user = just system prompt
    int sys_pos = 0;
    if (sys_pt && sys_pt->count > 0) {
        // Pre-embed all system prompt tokens
        float *sys_embed_batch = NULL;
        if (sys_pt->count > 1) {
            sys_embed_batch = malloc((size_t)sys_pt->count * cfg.hidden_dim * sizeof(float));
            for (int i = 0; i < sys_pt->count; i++) {
                embed_lookup(wf, sys_pt->ids[i], sys_embed_batch + (size_t)i * cfg.hidden_dim);
            }
        }
        // Intermediate system prompt tokens: discard last-layer expert output
        for (int i = 0; i < sys_pt->count - 1; i++) {
            cache_telemetry_note_token();
            if (sys_embed_batch) {
                memcpy(hidden, sys_embed_batch + (size_t)i * cfg.hidden_dim,
                       cfg.hidden_dim * sizeof(float));
            } else {
                embed_lookup(wf, sys_pt->ids[i], hidden);
            }
            for (int layer = 0; layer < cfg.num_layers; layer++) {
                int is_full = cfg.is_full_attn[layer];
                fused_layer_forward(wf, layer, hidden,
                                    is_full ? kv_caches[layer] : NULL,
                                    is_full ? NULL : layer_states[layer],
                                    sys_pos,
                                    layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                    K, layer_fds[layer]);
            }
            discard_deferred_experts();
            sys_pos++;
        }
        // Last system prompt token: full completion
        {
            cache_telemetry_note_token();
            if (sys_embed_batch) {
                memcpy(hidden, sys_embed_batch + (size_t)(sys_pt->count - 1) * cfg.hidden_dim,
                       cfg.hidden_dim * sizeof(float));
            } else {
                embed_lookup(wf, sys_pt->ids[0], hidden);
            }
            for (int layer = 0; layer < cfg.num_layers; layer++) {
                int is_full = cfg.is_full_attn[layer];
                fused_layer_forward(wf, layer, hidden,
                                    is_full ? kv_caches[layer] : NULL,
                                    is_full ? NULL : layer_states[layer],
                                    sys_pos,
                                    layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                    K, layer_fds[layer]);
            }
            complete_deferred_experts();
            sys_pos++;
        }
        if (sys_embed_batch) { free(sys_embed_batch); sys_embed_batch = NULL; }
        // Sync CPU state → GPU for delta-net
        sync_cpu_to_gpu_delta_state_serve(layer_states);
        fprintf(stderr, "[serve] System prompt cached: %d tokens prefilled\n", sys_pos);
    }
    free(sys_pt->ids);
    free(sys_pt);

    // Save snapshot of KV caches + linear attention state after system prompt
    // These are restored at the start of each request instead of resetting to zero
    typedef struct {
        float *k_snapshot;
        float *v_snapshot;
        int len;
    } KVSnapshot;
    KVSnapshot kv_snapshots[cfg.num_layers];
    memset(kv_snapshots, 0, sizeof(kv_snapshots));

    // Linear attention snapshots
    float *la_conv_snapshots[cfg.num_layers];
    float *la_ssm_snapshots[cfg.num_layers];
    memset(la_conv_snapshots, 0, sizeof(la_conv_snapshots));
    memset(la_ssm_snapshots, 0, sizeof(la_ssm_snapshots));

    size_t kv_dim = cfg.num_kv_heads * cfg.head_dim;
    size_t conv_state_size = (cfg.conv_kernel_size - 1) * cfg.linear_conv_dim * sizeof(float);
    size_t ssm_state_size = cfg.linear_num_v_heads * cfg.linear_value_dim * cfg.linear_key_dim * sizeof(float);

    for (int i = 0; i < cfg.num_layers; i++) {
        if (kv_caches[i]) {
            size_t sz = sys_pos * kv_dim * sizeof(float);
            kv_snapshots[i].k_snapshot = malloc(sz);
            kv_snapshots[i].v_snapshot = malloc(sz);
            memcpy(kv_snapshots[i].k_snapshot, kv_caches[i]->k_cache, sz);
            memcpy(kv_snapshots[i].v_snapshot, kv_caches[i]->v_cache, sz);
            kv_snapshots[i].len = kv_caches[i]->len;
        }
        if (layer_states[i]) {
            LinearAttnState *s = (LinearAttnState *)layer_states[i];
            la_conv_snapshots[i] = malloc(conv_state_size);
            la_ssm_snapshots[i] = malloc(ssm_state_size);
            memcpy(la_conv_snapshots[i], s->conv_state, conv_state_size);
            memcpy(la_ssm_snapshots[i], s->ssm_state, ssm_state_size);
        }
    }
    // Also snapshot GPU delta-net state
    void **gpu_delta_snapshots = calloc(cfg.num_linear_layers, sizeof(void *));
    void **gpu_conv_snapshots = calloc(cfg.num_linear_layers, sizeof(void *));
    // already zeroed by calloc
    // already zeroed by calloc
    if (g_metal && g_metal->delta_net_step) {
        for (int i = 0; i < cfg.num_linear_layers; i++) {
            if (g_metal->buf_delta_state[i]) {
                size_t sz = (size_t)cfg.linear_num_v_heads*cfg.linear_value_dim*cfg.linear_key_dim*sizeof(float);
                gpu_delta_snapshots[i] = malloc(sz);
                memcpy(gpu_delta_snapshots[i], [g_metal->buf_delta_state[i] contents], sz);
            }
            if (g_metal->buf_conv_state[i]) {
                size_t sz = (cfg.conv_kernel_size-1)*(size_t)cfg.linear_conv_dim*sizeof(float);
                gpu_conv_snapshots[i] = malloc(sz);
                memcpy(gpu_conv_snapshots[i], [g_metal->buf_conv_state[i] contents], sz);
            }
        }
    }
    int sys_prompt_len = sys_pos;  // number of tokens in system prompt cache

    // ---- Session state: track one active conversation session ----
    // The KV caches + linear attention state ARE the session.
    // We just track whether to restore from snapshot (new session) or continue (same session).
    char active_session_id[64] = {0};
    int session_pos = 0;  // RoPE position after last generation for the active session

    for (;;) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) { perror("accept"); continue; }

        // Read HTTP request
        char *reqbuf = malloc(1024 * 1024); // 1MB max request
        int reqlen = read_http_request(client_fd, reqbuf, 1024 * 1024);
        if (reqlen <= 0) { free(reqbuf); close(client_fd); continue; }

        // Parse method and path from first line
        char method[16] = {0}, path[256] = {0};
        sscanf(reqbuf, "%15s %255s", method, path);

        // Handle CORS preflight
        if (strcmp(method, "OPTIONS") == 0) {
            http_write_str(client_fd, CORS_RESPONSE);
            free(reqbuf); close(client_fd);
            continue;
        }

        // GET /health
        if (strcmp(method, "GET") == 0 && strcmp(path, "/health") == 0) {
            const char *resp =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: close\r\n"
                "\r\n"
                "{\"status\":\"ok\",\"model\":\"qwen3.5-35b-a3b\"}\n";
            http_write_str(client_fd, resp);
            free(reqbuf); close(client_fd);
            continue;
        }

        // GET /v1/models
        if (strcmp(method, "GET") == 0 && strcmp(path, "/v1/models") == 0) {
            const char *resp =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: close\r\n"
                "\r\n"
                "{\"object\":\"list\",\"data\":[{\"id\":\"qwen3.5-35b-a3b\","
                "\"object\":\"model\",\"owned_by\":\"local\"}]}\n";
            http_write_str(client_fd, resp);
            free(reqbuf); close(client_fd);
            continue;
        }

        // POST /v1/chat/completions
        if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/chat/completions") == 0) {
            // Find body (after \r\n\r\n)
            char *body = strstr(reqbuf, "\r\n\r\n");
            if (!body) {
                http_write_str(client_fd,
                    "HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n"
                    "{\"error\":\"no body\"}\n");
                free(reqbuf); close(client_fd); continue;
            }
            body += 4;

            // Extract session_id and max_tokens BEFORE content extraction
            // (extract_last_content mutates the body buffer in place)
            int max_gen = extract_max_tokens(body, 8192);
            if (max_gen > 32768) max_gen = 32768;
            char req_session_id[64] = {0};
            int has_session = extract_session_id(body, req_session_id, sizeof(req_session_id));

            // Extract user content from messages (mutates body — must be last)
            char *content = extract_last_content(body);
            if (!content || strlen(content) == 0) {
                http_write_str(client_fd,
                    "HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n"
                    "{\"error\":\"no content in messages\"}\n");
                free(reqbuf); close(client_fd); continue;
            }
            int is_continuation = (has_session &&
                                   active_session_id[0] != '\0' &&
                                   strcmp(req_session_id, active_session_id) == 0);

            // Session persistence is handled by the client (chat.m)

            char request_id[64];
            snprintf(request_id, sizeof(request_id), "chatcmpl-%llu", ++req_counter);

            fprintf(stderr, "[serve] %s content=%zu chars, max_tokens=%d, session=%s%s\n",
                    request_id, strlen(content), max_gen,
                    has_session ? req_session_id : "(none)",
                    is_continuation ? " [CONTINUE]" : " [NEW]");

            // ---- Tokenize ----
            // Continuation: prefix with <|im_end|>\n to close prior assistant turn
            // New session: just the user turn (system prompt restored from snapshot)
            PromptTokens *pt;
            if (is_continuation) {
                pt = tokenize_continuation_turn(content);
            } else {
                pt = tokenize_user_turn(content);
            }
            if (!pt) {
                http_write_str(client_fd,
                    "HTTP/1.1 500 Internal Server Error\r\nConnection: close\r\n\r\n"
                    "{\"error\":\"tokenization failed\"}\n");
                free(reqbuf); close(client_fd); continue;
            }

            fprintf(stderr, "[serve] %s prompt=%d tokens%s\n", request_id, pt->count,
                    is_continuation ? " (continuation — skipping snapshot restore)" : "");

            int pos;
            if (is_continuation) {
                // ---- Continue from existing session state ----
                // The KV caches + linear attention state already contain the full
                // conversation history. Just set pos to where we left off.
                pos = session_pos;
            } else {
                // ---- Restore state from system prompt snapshot ----
                // Instead of resetting to zero, restore to the cached system prompt state.
                // This skips re-prefilling the system prompt tokens (~20 tokens, ~6s saved).
                for (int i = 0; i < cfg.num_layers; i++) {
                    if (kv_caches[i] && kv_snapshots[i].k_snapshot) {
                        size_t sz = sys_prompt_len * kv_dim * sizeof(float);
                        memcpy(kv_caches[i]->k_cache, kv_snapshots[i].k_snapshot, sz);
                        memcpy(kv_caches[i]->v_cache, kv_snapshots[i].v_snapshot, sz);
                        kv_caches[i]->len = kv_snapshots[i].len;
                        // Also restore GPU KV mirror
                        if (g_metal) {
                            int fa_idx = cfg.full_attn_index[i];
                            if (fa_idx >= 0 && fa_idx < cfg.num_full_attn_layers) {
                                memcpy([g_metal->buf_kv_k[fa_idx] contents],
                                       kv_snapshots[i].k_snapshot, sz);
                                memcpy([g_metal->buf_kv_v[fa_idx] contents],
                                       kv_snapshots[i].v_snapshot, sz);
                            }
                        }
                    } else if (kv_caches[i]) {
                        kv_caches[i]->len = 0;
                    }
                    if (layer_states[i] && la_conv_snapshots[i]) {
                        LinearAttnState *s = (LinearAttnState *)layer_states[i];
                        memcpy(s->conv_state, la_conv_snapshots[i], conv_state_size);
                        memcpy(s->ssm_state, la_ssm_snapshots[i], ssm_state_size);
                    } else if (layer_states[i]) {
                        LinearAttnState *s = (LinearAttnState *)layer_states[i];
                        memset(s->conv_state, 0, conv_state_size);
                        memset(s->ssm_state, 0, ssm_state_size);
                    }
                }
                // Restore GPU delta-net state
                if (g_metal && g_metal->delta_net_step) {
                    for (int i = 0; i < cfg.num_linear_layers; i++) {
                        if (gpu_delta_snapshots[i] && g_metal->buf_delta_state[i])
                            memcpy([g_metal->buf_delta_state[i] contents],
                                   gpu_delta_snapshots[i], (size_t)cfg.linear_num_v_heads*cfg.linear_value_dim*cfg.linear_key_dim*sizeof(float));
                        if (gpu_conv_snapshots[i] && g_metal->buf_conv_state[i])
                            memcpy([g_metal->buf_conv_state[i] contents],
                                   gpu_conv_snapshots[i], (cfg.conv_kernel_size-1)*(size_t)cfg.linear_conv_dim*sizeof(float));
                    }
                } else {
                    reset_delta_net_state();
                }
                pos = sys_prompt_len;  // start after cached system prompt
                // Update active session
                if (has_session) {
                    strncpy(active_session_id, req_session_id, sizeof(active_session_id) - 1);
                    active_session_id[sizeof(active_session_id) - 1] = '\0';
                } else {
                    active_session_id[0] = '\0';
                }
            }
            if (g_cache_telemetry_enabled) cache_telemetry_reset();

            // ---- Send SSE headers ----
            http_write_str(client_fd, SSE_HEADERS);

            // ---- Batch prefill ----
            double t_prefill = now_ms();
            // Pre-embed all request tokens
            float *serve_embed_batch = NULL;
            if (pt->count > 1) {
                serve_embed_batch = malloc((size_t)pt->count * cfg.hidden_dim * sizeof(float));
                for (int i = 0; i < pt->count; i++) {
                    embed_lookup(wf, pt->ids[i], serve_embed_batch + (size_t)i * cfg.hidden_dim);
                }
            }
            // Intermediate prefill tokens: discard last-layer expert output
            for (int i = 0; i < pt->count - 1; i++) {
                cache_telemetry_note_token();
                if (serve_embed_batch) {
                    memcpy(hidden, serve_embed_batch + (size_t)i * cfg.hidden_dim,
                           cfg.hidden_dim * sizeof(float));
                } else {
                    embed_lookup(wf, pt->ids[i], hidden);
                }
                for (int layer = 0; layer < cfg.num_layers; layer++) {
                    int is_full = cfg.is_full_attn[layer];
                    fused_layer_forward(wf, layer, hidden,
                                        is_full ? kv_caches[layer] : NULL,
                                        is_full ? NULL : layer_states[layer],
                                        pos,
                                        layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                        K, layer_fds[layer]);
                }
                discard_deferred_experts();
                pos++;
            }
            // Last prefill token: full completion (need hidden for logits)
            {
                cache_telemetry_note_token();
                if (serve_embed_batch) {
                    memcpy(hidden, serve_embed_batch + (size_t)(pt->count - 1) * cfg.hidden_dim,
                           cfg.hidden_dim * sizeof(float));
                } else {
                    embed_lookup(wf, pt->ids[0], hidden);
                }
                for (int layer = 0; layer < cfg.num_layers; layer++) {
                    int is_full = cfg.is_full_attn[layer];
                    fused_layer_forward(wf, layer, hidden,
                                        is_full ? kv_caches[layer] : NULL,
                                        is_full ? NULL : layer_states[layer],
                                        pos,
                                        layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                        K, layer_fds[layer]);
                }
                complete_deferred_experts();
                pos++;
            }
            if (serve_embed_batch) { free(serve_embed_batch); serve_embed_batch = NULL; }
            double prefill_ms = now_ms() - t_prefill;
            fprintf(stderr, "[serve] %s prefill=%d tokens in %.0fms\n",
                    request_id, pt->count, prefill_ms);

            // ---- Final norm + LM head for first token ----
            if (final_norm_w) {
                float *normed = malloc(cfg.hidden_dim * sizeof(float));
                cpu_rms_norm(hidden, final_norm_w, normed, cfg.hidden_dim, cfg.rms_norm_eps);
                memcpy(hidden, normed, cfg.hidden_dim * sizeof(float));
                free(normed);
            }
            lm_head_forward(wf, hidden, logits);
            int next_token = cpu_argmax(logits, cfg.vocab_size);

            // ---- Auto-regressive generation with SSE streaming ----
            if (g_pred_enabled) {
                g_pred_generating = 1;
                g_pred_valid = 0;
            }
            double t_gen = now_ms();
            int gen_count = 0;
            int in_think = 0;
            int think_tokens = 0;
            // Accumulate response for session persistence
            char *gen_response = calloc(1, 256 * 1024);
            int gen_resp_len = 0;

            // UTF-8 streaming buffer: ensures we only emit complete codepoints
            ServeUtf8Buf u8buf = {0};

            for (int gen = 0; gen < max_gen; gen++) {
                if (next_token == cfg.eos_token_ids[0] || next_token == cfg.eos_token_ids[1]) {
                    // Feed EOS through the model so session state includes it
                    cache_telemetry_note_token();
                    embed_lookup(wf, next_token, hidden);
                    for (int layer = 0; layer < cfg.num_layers; layer++) {
                        int is_full = cfg.is_full_attn[layer];
                        fused_layer_forward(wf, layer, hidden,
                                            is_full ? kv_caches[layer] : NULL,
                                            is_full ? NULL : layer_states[layer],
                                            pos,
                                            layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                            K, layer_fds[layer]);
                    }
                    discard_deferred_experts();
                    pos++;
                    break;
                }

                // Think budget enforcement
                if (next_token == cfg.think_start_token) in_think = 1;
                if (next_token == cfg.think_end_token) in_think = 0;
                if (in_think) {
                    think_tokens++;
                    if (g_think_budget > 0 && think_tokens >= g_think_budget) {
                        next_token = cfg.think_end_token;  // force end thinking
                        in_think = 0;
                    }
                }

                const char *tok_str = decode_token(vocab, next_token);
                // Accumulate non-thinking response for session persistence
                if (!in_think && tok_str && gen_resp_len + (int)strlen(tok_str) < 256*1024 - 1) {
                    int tlen = (int)strlen(tok_str);
                    memcpy(gen_response + gen_resp_len, tok_str, tlen);
                    gen_resp_len += tlen;
                    gen_response[gen_resp_len] = 0;
                }
                // Push token through UTF-8 buffer to ensure complete codepoints
                {
                    char u8out[4096];
                    int tok_len = tok_str ? (int)strlen(tok_str) : 0;
                    const char *safe = serve_utf8_push(&u8buf, tok_str, tok_len,
                                                       u8out, sizeof(u8out));
                    if (safe[0]) {
                        if (sse_send_delta(client_fd, request_id, safe) < 0) {
                            fprintf(stderr, "[serve] %s client disconnected, stopping generation\n", request_id);
                            break;
                        }
                    }
                }
                gen_count++;

                // Generate next
                cache_telemetry_note_token();
                embed_lookup(wf, next_token, hidden);
                for (int layer = 0; layer < cfg.num_layers; layer++) {
                    int is_full = cfg.is_full_attn[layer];
                    fused_layer_forward(wf, layer, hidden,
                                        is_full ? kv_caches[layer] : NULL,
                                        is_full ? NULL : layer_states[layer],
                                        pos,
                                        layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                        K, layer_fds[layer]);
                }
                complete_deferred_experts();
                pos++;

                if (final_norm_w) {
                    float *normed = malloc(cfg.hidden_dim * sizeof(float));
                    cpu_rms_norm(hidden, final_norm_w, normed, cfg.hidden_dim, cfg.rms_norm_eps);
                    memcpy(hidden, normed, cfg.hidden_dim * sizeof(float));
                    free(normed);
                }
                lm_head_forward(wf, hidden, logits);
                next_token = cpu_argmax(logits, cfg.vocab_size);
            }

            // Flush any remaining incomplete UTF-8 bytes before done marker
            {
                char u8out[8];
                const char *safe = serve_utf8_flush(&u8buf, u8out, sizeof(u8out));
                if (safe[0]) sse_send_delta(client_fd, request_id, safe);
            }

            sse_send_done(client_fd, request_id);

            // ---- Save session state ----
            free(gen_response);
            // The KV caches + linear attention state already contain this conversation.
            // Just record the position so the next request can continue from here.
            session_pos = pos;
            fprintf(stderr, "[serve] %s session_pos=%d (session=%s)\n",
                    request_id, session_pos,
                    active_session_id[0] ? active_session_id : "(none)");

            double gen_ms = now_ms() - t_gen;
            fprintf(stderr, "[serve] %s generated=%d tokens in %.0fms (%.2f tok/s)\n",
                    request_id, gen_count, gen_ms,
                    gen_count > 0 ? gen_count * 1000.0 / gen_ms : 0.0);
            if (g_expert_cache) {
                cache_telemetry_print(g_expert_cache->hits, g_expert_cache->misses);
            } else if (g_malloc_cache) {
                cache_telemetry_print(g_malloc_cache->hits, g_malloc_cache->misses);
            }

            free(pt->ids);
            free(pt);
            free(reqbuf);
            close(client_fd);
            continue;
        }

        // Unknown endpoint
        const char *resp404 =
            "HTTP/1.1 404 Not Found\r\n"
            "Content-Type: application/json\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Connection: close\r\n"
            "\r\n"
            "{\"error\":\"not found\"}\n";
        http_write_str(client_fd, resp404);
        free(reqbuf);
        close(client_fd);
    }
}

// ============================================================================

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --model PATH         Model path\n");
    printf("  --weights PATH       model_weights.bin path\n");
    printf("  --manifest PATH      model_weights.json path\n");
    printf("  --vocab PATH         vocab.bin path\n");
    printf("  --prompt-tokens PATH prompt_tokens.bin path\n");
    printf("  --prompt TEXT         Prompt text (requires encode_prompt.py)\n");
    printf("  --tokens N           Max tokens to generate (default: 20)\n");
    printf("  --k N                Active experts per layer (default: 4)\n");
    printf("  --cache-entries N    Expert LRU cache size (default: 2500, 0 = disabled)\n");
    printf("  --malloc-cache N     Malloc expert cache entries (e.g., 2581 = 17GB for 80%% hit)\n");
    printf("  --cpu-linear         Disable fused GPU delta-net and use the older CPU/hybrid linear path\n");
    printf("  --timing             Enable per-layer timing breakdown\n");
    printf("  --freq               Enable expert frequency tracking + analysis\n");
    printf("  --cache-telemetry    Report cold vs eviction misses and reuse distance\n");
    printf("  --2bit               Use 2-bit quantized experts (packed_experts_2bit/)\n");
    printf("  --tiered             Use tiered quantization: hot=4-bit, cold=2-bit (packed_experts_tiered/)\n");
    printf("  --gpu-linear         Alias for the fused GPU delta-net path (default)\n");
    printf("  --predict            Enable temporal expert prediction (prefetch during CMD1_wait)\n");
    printf("  --collect-routing F  Log routing data to binary file F (for predictor training)\n");
    printf("  --think-budget N     Max thinking tokens before force </think> (default: 2048, 0=unlimited)\n");
    printf("  --ppl PATH           Measure perplexity on ground truth token file\n");
    printf("  --stream             Clean streaming output (no progress, no stats)\n");
    printf("  --pfb N              Enable batched prefill\n");
    printf("  --prefill-skip-experts  Skip routed experts for intermediate prefill tokens (shared expert only)\n");
    printf("  --no-batched-linear  Disable batched linear-attention prefill kernels (uses routed MoE tail + batched full-attn)\n");
    printf("  --gguf-embedding P   Use extracted GGUF Q8_0 embedding blob\n");
    printf("  --nax                Enable NAX tensor matmul for LM head (Metal 4+, SLOWER for M=1 decode)\n");
    printf("  --no-nax             Disable NAX (default)\n");
    printf("  --serve PORT         Run HTTP server (OpenAI-compatible API)\n");
    printf("  --fp8                Use FP8 E4M3 KV cache (4x memory reduction)\n");
    printf("  --fused-attn         Enable fused online softmax attention (experimental)\n");
    printf("  --fp16               Use FP16 accumulation in dequant kernels (experimental)\n");
    printf("  --fused-expert       Enable fused gate+up+SwiGLU expert kernel\n");
    printf("  --no-fused-expert    Disable fused gate+up+SwiGLU expert kernel\n");
    printf("  --no-cmd-merge       Disable CMD1+CMD2 merge for linear attention\n");
    printf("  --expert-prefetch    Enable cross-layer expert prefetch\n");
    printf("  --sliding-window N   Sliding window size for full attention (circular KV cache, 0=unlimited)\n");
    printf("  --h2o N              H2O KV cache budget (sinks + recent + heavy hitters, 0=disabled)\n");
    printf("  --h2o-sinks N        Number of attention sink tokens (default: 4)\n");
    printf("  --help               This message\n");
}

// Batched prefill implementation (separate file for maintainability)
#include "batched_prefill.h"

#ifndef INFER_LIB_MODE
int main(int argc, char **argv) {
    @autoreleasepool {
        const char *model_path = getenv("FLASH_MOE_MODEL");
        const char *weights_path = NULL;
        const char *manifest_path = NULL;
        const char *vocab_path = NULL;
        const char *prompt_tokens_path = NULL;
        const char *prompt_text = NULL;
        const char *ppl_tokens_path = NULL;
        int max_tokens = 20;
        int K = 8;
        int K_explicit = 0;  // set to 1 if --k was passed
        int cache_entries = 0;  // default 0: trust OS page cache (38% faster than Metal LRU)
        int malloc_cache_entries = 0;  // 0 = disabled (override with --malloc-cache)
        int serve_port = 0;  // 0 = disabled, >0 = HTTP serve mode

        static struct option long_options[] = {
            {"model",         required_argument, 0, 'm'},
            {"weights",       required_argument, 0, 'w'},
            {"manifest",      required_argument, 0, 'j'},
            {"vocab",         required_argument, 0, 'v'},
            {"prompt-tokens", required_argument, 0, 'p'},
            {"prompt",        required_argument, 0, 'P'},
            {"tokens",        required_argument, 0, 't'},
            {"k",             required_argument, 0, 'k'},
            {"cache-entries",  required_argument, 0, 'C'},
            {"malloc-cache",   required_argument, 0, 'M'},
            {"cpu-linear",    no_argument,       0, 'L'},
            {"skip-linear",   no_argument,       0, 'S'},
            {"timing",        no_argument,       0, 'T'},
            {"freq",          no_argument,       0, 'F'},
            {"cache-telemetry", no_argument,     0, 'E'},
            {"2bit",          no_argument,       0, '2'},
            {"tiered",        no_argument,       0, 'Q'},
            {"gpu-linear",    no_argument,       0, 'G'},
            {"think-budget",  required_argument, 0, 'B'},
            {"serve",         required_argument, 0, 'R'},
            {"predict",       no_argument,       0, 'D'},
            {"collect-routing", required_argument, 0, 'Z'},
            {"fp8",           no_argument,       0, 1001},
            {"fused-attn",    no_argument,       0, 1002},
            {"fp16",          no_argument,       0, 1003},
            {"fused-expert",  no_argument,       0, 1007},
            {"no-fused-expert", no_argument,     0, 1004},
            {"cmd-merge",     no_argument,       0, 1008},
            {"no-cmd-merge",  no_argument,       0, 1005},
            {"expert-prefetch", no_argument,     0, 1006},
            {"sliding-window", required_argument, 0, 1009},
            {"h2o",            required_argument, 0, 1010},
            {"h2o-sinks",      required_argument, 0, 1011},
            {"ppl",           required_argument, 0, 905},
            {"stream",        no_argument,       0, 'O'},
            {"nax",           no_argument,       0, 'X'},
            {"no-nax",        no_argument,       0, 'x'},
            {"pfb",           required_argument, 0, 900},
            {"prefill-skip-experts", no_argument, 0, 901},
            {"prefill-k",     required_argument, 0, 903},
            {"prefill-experts-full-only", no_argument, 0, 904},
            {"no-batched-linear", no_argument, 0, 902},
            {"help",          no_argument,       0, 'h'},
            {0, 0, 0, 0}
        };

        int c;
        while ((c = getopt_long(argc, argv, "m:w:j:v:p:P:t:k:C:M:R:B:LSTFE2Gh", long_options, NULL)) != -1) {
            switch (c) {
                case 'm': model_path = optarg; break;
                case 'w': weights_path = optarg; break;
                case 'j': manifest_path = optarg; break;
                case 'v': vocab_path = optarg; break;
                case 'p': prompt_tokens_path = optarg; break;
                case 'P': prompt_text = optarg; break;
                case 't': max_tokens = atoi(optarg); break;
                case 'k': K = atoi(optarg); K_explicit = 1; break;
                case 'C': cache_entries = atoi(optarg); break;
                case 'M': malloc_cache_entries = atoi(optarg); break;
                case 'L': gpu_linear_attn_enabled = 0; break;
                case 'S': linear_attn_bypass = 1; break;
                case 'T': g_timing_enabled = 1; break;
                case 'F': g_freq_tracking = 1; break;
                case 'E': g_cache_telemetry_enabled = 1; break;
                case '2': g_use_2bit = 1; break;
                case 'Q': g_use_tiered = 1; break;
                case 'G': gpu_linear_attn_enabled = 1; break;
                case 'D': g_pred_enabled = 1; break;
                case 'Z':
                    g_routing_log = fopen(optarg, "wb");
                    if (!g_routing_log) {
                        fprintf(stderr, "ERROR: cannot open routing log: %s\n", optarg);
                        return 1;
                    }
                    break;
                case 'B': g_think_budget = atoi(optarg); break;
                case 'R': serve_port = atoi(optarg); break;
                case 1001: g_use_fp8_kv = 1; break;
                case 1002: g_fused_attention_enabled = 1; break;
                case 1003: g_use_fp16_accum = 1; break;
                case 1004: g_fused_expert_enabled = 0; break;
                case 1005: g_cmd_merge_enabled = 0; break;
                case 1006: g_expert_prefetch_enabled = 1; break;
                case 1007: g_fused_expert_enabled = 1; break;
                case 1008: g_cmd_merge_enabled = 1; break;
                case 1009: g_sliding_window = atoi(optarg); break;
                case 1010: g_h2o_budget = atoi(optarg); break;
                case 1011: g_h2o_num_sinks = atoi(optarg); break;
                case 905: ppl_tokens_path = optarg; break;
                case 'O': g_stream_mode = 1; break;
                case 'X': g_nax_disabled = 0; break;  // --nax: enable
                case 'x': g_nax_disabled = 1; break;  // --no-nax: disable
                case 900: g_prefill_batch = atoi(optarg);
                    if (g_prefill_batch < 1) g_prefill_batch = 1;
                    if (g_prefill_batch > MAX_PFB) g_prefill_batch = MAX_PFB;
                    break;
                case 901: g_prefill_skip_experts = 1; break;
                case 903: g_prefill_k = atoi(optarg); break;
                case 904: g_prefill_experts_full_only = 1; break;
                case 902: g_disable_batched_linear = 1; break;
                case 'h': print_usage(argv[0]); return 0;
                default:  print_usage(argv[0]); return 1;
            }
        }

        // ---- Load model configuration from HF config.json ----
        load_model_config(model_path ? model_path : "");
        alloc_tracking_arrays();
        g_deferred.h_mid = calloc(cfg.hidden_dim, sizeof(float));

        // Cap K to MAX_K (buffer overflow safety)
        if (K > MAX_K) {
            fprintf(stderr, "WARNING: K=%d exceeds MAX_K=%d, capping to %d\n", K, MAX_K, MAX_K);
            K = MAX_K;
        }

        // Build default paths — check model directory first, then relative paths
        char default_weights[1024] = {0}, default_manifest[1024] = {0}, default_vocab[1024] = {0};

        if (!weights_path) {
            // 1. Try <model_path>/model_weights.bin
            if (model_path) {
                snprintf(default_weights, sizeof(default_weights),
                         "%s/model_weights.bin", model_path);
                if (access(default_weights, R_OK) != 0)
                    default_weights[0] = '\0';
            }
            // 2. Try relative paths
            if (!default_weights[0]) {
                snprintf(default_weights, sizeof(default_weights),
                         "metal_infer/model_weights.bin");
                if (access(default_weights, R_OK) != 0) {
                    snprintf(default_weights, sizeof(default_weights),
                             "model_weights.bin");
                }
            }
            weights_path = default_weights;
        }
        if (!manifest_path) {
            if (model_path) {
                snprintf(default_manifest, sizeof(default_manifest),
                         "%s/model_weights.json", model_path);
                if (access(default_manifest, R_OK) != 0)
                    default_manifest[0] = '\0';
            }
            if (!default_manifest[0]) {
                snprintf(default_manifest, sizeof(default_manifest),
                         "metal_infer/model_weights.json");
                if (access(default_manifest, R_OK) != 0) {
                    snprintf(default_manifest, sizeof(default_manifest),
                             "model_weights.json");
                }
            }
            manifest_path = default_manifest;
        }
        if (!vocab_path) {
            if (model_path) {
                snprintf(default_vocab, sizeof(default_vocab),
                         "%s/vocab.bin", model_path);
                if (access(default_vocab, R_OK) != 0)
                    default_vocab[0] = '\0';
            }
            if (!default_vocab[0]) {
                snprintf(default_vocab, sizeof(default_vocab),
                         "metal_infer/vocab.bin");
                if (access(default_vocab, R_OK) != 0) {
                    snprintf(default_vocab, sizeof(default_vocab),
                             "vocab.bin");
                }
            }
            vocab_path = default_vocab;
        }

        // Update K from config (unless explicitly overridden via --k)
        // The 397B model uses K=4 (actual active) despite num_experts_per_tok=10 in config
        // Other models (35B) use K=8. Auto-set K from config, capped to MAX_K.
        {
            int config_k = cfg.num_experts_per_tok;
            if (config_k > MAX_K) config_k = MAX_K;
            // Only override if user didn't explicitly set K via --k
            // (we detect this by checking if K is still the default 4)
            if (!K_explicit) {
                K = config_k;
                fprintf(stderr, "[config] K auto-set to %d from config (use --k N to override)\n", K);
            } else {
                fprintf(stderr, "[config] K override: %d (model default: %d)\n", K, cfg.num_experts_per_tok);
            }
        }

        // ---- Initialize Metal ----
        g_metal = metal_setup();
        if (!g_metal) {
            fprintf(stderr, "WARNING: Metal init failed, falling back to CPU\n");
        }

        // ---- Initialize persistent I/O thread pool ----
        io_pool_init();

        // ---- Initialize malloc expert cache (if requested) ----
        if (malloc_cache_entries > 0) {
            g_malloc_cache = malloc_cache_init(malloc_cache_entries, g_metal ? g_metal->device : MTLCreateSystemDefaultDevice());
            cache_entries = 0;  // disable Metal LRU cache when malloc cache is active
        }

        // ---- Initialize expert LRU cache ----
        if (cache_entries > 0 && g_metal) {
            g_expert_cache = expert_cache_new(g_metal->device, cache_entries);
        }

        printf("=== Flash-MoE Metal Inference Engine ===\n");
        printf("Config:   %s/config.json\n", cfg.model_path);
        printf("Model:    %s\n", model_path);
        printf("Weights:  %s\n", weights_path);
        printf("Manifest: %s\n", manifest_path);
        printf("Vocab:    %s\n", vocab_path);
        printf("K:        %d experts/layer\n", K);
        printf("Quant:    %s\n",
               g_use_tiered ? "tiered (hot=4-bit, cold=2-bit)" :
               g_use_2bit ? "2-bit experts" :
               "4-bit experts");
        printf("Linear:   %s\n", gpu_linear_attn_enabled ? "fused GPU delta-net" : "CPU/hybrid fallback");
        printf("Tokens:   %d\n", max_tokens);
        if (g_malloc_cache) {
            printf("Cache:    malloc %d entries (%.1f GB)\n",
                   malloc_cache_entries, (double)malloc_cache_entries * active_expert_size() / 1e9);
        } else {
            printf("Cache:    %d entries%s\n", cache_entries,
                   cache_entries > 0 ? "" : " (disabled)");
        }

        double t0 = now_ms();

        // ---- Load weights ----
        WeightFile *wf = open_weights(weights_path, manifest_path);
        if (!wf) {
            fprintf(stderr, "ERROR: Failed to load weights\n");
            return 1;
        }

        // Wrap weight file for Metal GPU access
        if (g_metal) {
            metal_set_weights(g_metal, wf->data, wf->size);
        }

        // ---- Load vocabulary ----
        Vocabulary *vocab = load_vocab(vocab_path);
        if (!vocab) {
            fprintf(stderr, "ERROR: Failed to load vocabulary\n");
            return 1;
        }

        // ---- Get prompt tokens (skip in serve mode) ----
        PromptTokens *pt = NULL;
        if (serve_port == 0) {
            if (prompt_text) {
                pt = encode_prompt_text_to_tokens(prompt_text);
                if (!pt) {
                    fprintf(stderr, "ERROR: Failed to encode prompt. Make sure encode_prompt.py exists.\n");
                    return 1;
                }
            } else if (!prompt_tokens_path) {
                pt = encode_prompt_text_to_tokens("Hello, what is");
                if (!pt) {
                    fprintf(stderr, "ERROR: No prompt tokens and encode_prompt.py not found\n");
                    return 1;
                }
            } else {
                pt = load_prompt_tokens(prompt_tokens_path);
            }

            if (!pt) {
                fprintf(stderr, "ERROR: Failed to load prompt tokens from %s\n", prompt_tokens_path);
                return 1;
            }
            printf("[prompt] %d tokens:", pt->count);
            for (int i = 0; i < pt->count && i < 20; i++) {
                printf(" %d", pt->ids[i]);
            }
            printf("\n");
        }

        // ---- Mutual exclusion: --tiered and --2bit cannot coexist ----
        if (g_use_tiered && g_use_2bit) {
            fprintf(stderr, "ERROR: --tiered and --2bit are mutually exclusive\n");
            exit(1);
        }

        // ---- Auto-detect tiered experts (takes priority over 2-bit auto-detect) ----
        if (!g_use_2bit && !g_use_tiered) {
            char probe[1024];
            snprintf(probe, sizeof(probe), "%s/packed_experts_tiered/tiered_manifest.json", model_path);
            if (access(probe, F_OK) == 0) {
                if (load_tiered_manifest(model_path)) {
                    g_use_tiered = 1;
                }
            }
        }

        // ---- Load tiered manifest if --tiered was explicitly set ----
        if (g_use_tiered && !g_tiered_manifest) {
            if (!load_tiered_manifest(model_path)) {
                fprintf(stderr, "ERROR: --tiered specified but no tiered_manifest.json found in %s/packed_experts_tiered/\n", model_path);
                exit(1);
            }
        }

        // ---- Auto-detect 2-bit experts ----
        if (!g_use_2bit && !g_use_tiered) {
            char probe[1024];
            snprintf(probe, sizeof(probe), "%s/packed_experts_2bit/layer_00.bin", model_path);
            int pfd = open(probe, O_RDONLY);
            if (pfd >= 0) {
                close(pfd);
                snprintf(probe, sizeof(probe), "%s/packed_experts/layer_00.bin", model_path);
                int pfd4 = open(probe, O_RDONLY);
                if (pfd4 < 0) {
                    g_use_2bit = 1;
                    printf("[auto] Using 2-bit experts (4-bit not found)\n");
                } else {
                    close(pfd4);
                }
            }
        }

        // ---- Open + mmap packed expert files ----
        // Tiered I/O: two fds per layer file.
        //   layer_fds[i]      = warm fd (page cached) — for experts seen before
        //   layer_fds_cold[i] = cold fd (F_NOCACHE)   — for first-time expert reads
        // Seen-expert bitset tracks which (layer, expert) pairs have been read before.
        // First read goes through cold fd (no page cache pollution).
        // Subsequent reads go through warm fd (page cache hit = 32 GB/s vs 5.5 GB/s).
        int *layer_fds = calloc(cfg.num_layers, sizeof(int));
        int *layer_fds_cold = calloc(cfg.num_layers, sizeof(int));
        void **layer_mmaps = calloc(cfg.num_layers, sizeof(void *));
        size_t *layer_mmap_sizes = calloc(cfg.num_layers, sizeof(size_t));
        int expert_layers_available = 0;
        int q3_layout_loaded = 0;

        // Reset the global seen-expert bitset
        memset(g_expert_seen, 0, cfg.num_layers * ((cfg.num_experts + 7) / 8));

        int layers_2bit = 0, layers_q3 = 0, layers_q3_outlier = 0, layers_4bit = 0;
        memset(g_layer_is_2bit, 0, sizeof(g_layer_is_2bit));
        memset(g_layer_is_q3_hybrid, 0, sizeof(g_layer_is_q3_hybrid));
        memset(g_layer_is_q3_outlier, 0, sizeof(g_layer_is_q3_outlier));
        if (g_use_q3_experts) {
            q3_layout_loaded = load_q3_layout_manifest(model_path);
        }

        for (int i = 0; i < cfg.num_layers; i++) {
            char path[1024];
            layer_fds[i] = -1;

            if (g_use_tiered) {
                snprintf(path, sizeof(path), "%s/packed_experts_tiered/layer_%02d.bin", model_path, i);
                layer_fds[i] = open(path, O_RDONLY);
            } else if (g_use_q3_experts) {
                snprintf(path, sizeof(path), "%s/packed_experts_Q3/layer_%02d.bin", model_path, i);
                layer_fds[i] = open(path, O_RDONLY);
                if (layer_fds[i] >= 0) {
                    if (q3_layout_loaded && g_q3_layer_layout_valid[i]) {
                        if (expert_layout_is_q3_outlier(&g_q3_layer_layouts[i])) {
                            g_layer_is_q3_outlier[i] = 1;
                            layers_q3_outlier++;
                        } else {
                            g_layer_is_q3_hybrid[i] = 1;
                            layers_q3++;
                        }
                    } else if (i == Q3_OUTLIER_LAYER) {
                        g_layer_is_q3_outlier[i] = 1;
                        layers_q3_outlier++;
                    } else {
                        g_layer_is_q3_hybrid[i] = 1;
                        layers_q3++;
                    }
                } else {
                    snprintf(path, sizeof(path), "%s/packed_experts/layer_%02d.bin", model_path, i);
                    layer_fds[i] = open(path, O_RDONLY);
                    if (layer_fds[i] >= 0) {
                        g_layer_is_q3_hybrid[i] = 0;
                        layers_4bit++;
                    }
                }
            } else if (g_use_2bit) {
                // Try 2-bit first
                snprintf(path, sizeof(path), "%s/packed_experts_2bit/layer_%02d.bin", model_path, i);
                layer_fds[i] = open(path, O_RDONLY);
                if (layer_fds[i] >= 0) {
                    g_layer_is_2bit[i] = 1;
                    layers_2bit++;
                } else {
                    // Fall back to 4-bit for this layer
                    snprintf(path, sizeof(path), "%s/packed_experts/layer_%02d.bin", model_path, i);
                    layer_fds[i] = open(path, O_RDONLY);
                    if (layer_fds[i] >= 0) {
                        g_layer_is_2bit[i] = 0;
                        layers_4bit++;
                    }
                }
            } else {
                snprintf(path, sizeof(path), "%s/packed_experts/layer_%02d.bin", model_path, i);
                layer_fds[i] = open(path, O_RDONLY);
            }

            layer_fds_cold[i] = -1;  // no longer used (trust OS page cache)
            layer_mmaps[i] = MAP_FAILED;
            layer_mmap_sizes[i] = 0;
            if (layer_fds[i] >= 0) {
                expert_layers_available++;
                // Disable readahead: expert reads are random (different offsets per token).
                // Read-ahead prefetches adjacent data we won't use, wasting SSD bandwidth.
                fcntl(layer_fds[i], F_RDAHEAD, 0);
                struct stat st;
                if (fstat(layer_fds[i], &st) == 0 && st.st_size > 0) {
#if TARGET_OS_IPHONE || TARGET_OS_IOS
                    // iOS: never mmap expert files. The async pread path (GCD dispatch_group)
                    // doesn't use mmap, and mmap'ing ~18GB+ of expert layer files exhausts
                    // iOS virtual address space (limited even with extended-virtual-addressing).
                    (void)st;
#else
                    if (g_cache_io_split <= 1) {
                        // macOS: mmap when fanout is disabled. With cache-io-split the
                        // pread fanout path is used exclusively and mmap just wastes
                        // virtual address space and adds VM overhead.
                        layer_mmaps[i] = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, layer_fds[i], 0);
                    }
#endif
                    if (layer_mmaps[i] != MAP_FAILED) {
                        layer_mmap_sizes[i] = st.st_size;
                    }
                }
            }
        }
        const char *io_mode;
#if TARGET_OS_IPHONE || TARGET_OS_IOS
        io_mode = g_cache_io_split > 1 ? "pread fanout" : "pread (no mmap)";
#else
        io_mode = g_cache_io_split > 1 ? "pread fanout" : "mmap'd";
#endif
        printf("[experts] %d/%d packed layer files available (%s)\n",
               expert_layers_available, cfg.num_layers, io_mode);
        if (g_cache_io_split > 1) {
            printf("[fanout] cache-io-split=%d → %d page-aligned chunks per expert\n",
                   g_cache_io_split, active_cache_io_split(active_expert_size()));
        }

        // ---- LZ4 compressed experts: auto-detect and load ----
        {
            char lz4_probe[1024];
            snprintf(lz4_probe, sizeof(lz4_probe), "%s/packed_experts_lz4/layer_00.bin", model_path);
            if (!g_use_2bit && access(lz4_probe, R_OK) == 0) {
                int lz4_layers = 0;
                for (int i = 0; i < cfg.num_layers; i++) {
                    char lz4_path[1024];
                    snprintf(lz4_path, sizeof(lz4_path), "%s/packed_experts_lz4/layer_%02d.bin", model_path, i);
                    int lz4_fd = open(lz4_path, O_RDONLY);
                    if (lz4_fd >= 0) {
                        // Load index header (cfg.num_experts entries × 16 bytes)
                        g_lz4_index[i] = malloc(cfg.num_experts * sizeof(LZ4IndexEntry));
                        ssize_t nr = pread(lz4_fd, g_lz4_index[i],
                                           cfg.num_experts * sizeof(LZ4IndexEntry), 0);
                        if (nr == cfg.num_experts * (ssize_t)sizeof(LZ4IndexEntry)) {
                            // Replace the raw fd with the LZ4 fd
                            close(layer_fds[i]);
                            layer_fds[i] = lz4_fd;
                            fcntl(lz4_fd, F_RDAHEAD, 1);
                            lz4_layers++;
                        } else {
                            free(g_lz4_index[i]);
                            g_lz4_index[i] = NULL;
                            close(lz4_fd);
                        }
                    }
                }
                if (lz4_layers > 0) {
                    g_use_lz4 = 1;
                    // Allocate compressed read buffers (one per expert slot)
                    for (int k = 0; k < MAX_K; k++) {
                        g_lz4_comp_bufs[k] = malloc(cfg.expert_size_4bit + 4096);
                    }
                    printf("[lz4] %d/%d layers using LZ4 compressed experts\n",
                           lz4_layers, cfg.num_layers);
                }
            }
        }

        // Wire up tiered I/O globals
        g_layer_fds_cold = layer_fds_cold;
        if (!g_use_lz4)
            printf("[tiered-io] Cold fds (F_NOCACHE) + warm fds (page cached) active\n");

        // Warm page cache hint
        if (expert_layers_available > 0) {
            double t_warm = now_ms();
            for (int i = 0; i < cfg.num_layers; i++) {
                if (layer_fds[i] >= 0) {
                    char dummy[4096];
                    pread(layer_fds[i], dummy, sizeof(dummy), 0);
                }
            }
            printf("[warmup] Page cache hint: %.1f ms\n", now_ms() - t_warm);
        }

        // ---- Allocate per-layer state ----
        void **layer_states = calloc(cfg.num_layers, sizeof(void *));
        KVCache **kv_caches = calloc(cfg.num_layers, sizeof(KVCache *));

        for (int i = 0; i < cfg.num_layers; i++) {
            int is_full = cfg.is_full_attn[i];
            if (is_full) {
                kv_caches[i] = kv_cache_new();
            } else {
                layer_states[i] = linear_attn_state_new();
            }
        }

        double t_init = now_ms();
        printf("[init] Setup: %.1f ms\n\n", t_init - t0);

        // ---- Allocate working buffers ----
        float *hidden = calloc(cfg.hidden_dim, sizeof(float));
        float *logits = calloc(cfg.vocab_size, sizeof(float));
        uint16_t *final_norm_w = get_tensor_ptr(wf, "model.norm.weight");

        // ---- Serve mode: enter HTTP server loop (never returns) ----
        if (serve_port > 0) {
            reset_delta_net_state();
            serve_loop(serve_port, wf, vocab,
                       layer_states, kv_caches,
                       (void **)layer_mmaps, layer_fds,
                       hidden, logits, final_norm_w, K);
            // serve_loop never returns, but cleanup just in case
            free(hidden); free(logits);
            return 0;
        }

        // ---- Generate tokens ----
        reset_delta_net_state();  // zero GPU delta-net state before generation
        if (g_cache_telemetry_enabled) cache_telemetry_reset();
        printf("--- Generating %d tokens ---\n", max_tokens);
        int pos = 0;  // position counter for RoPE

        // ---- Batch prefill: pre-embed all prompt tokens ----
        // Embedding all tokens upfront into a batch buffer avoids interleaving
        // embed_lookup with GPU work, and enables the optimized prefill loop below.
        float *embed_batch = NULL;
        if (pt->count > 1) {
            embed_batch = malloc((size_t)pt->count * cfg.hidden_dim * sizeof(float));
            double t_embed = now_ms();
            for (int i = 0; i < pt->count; i++) {
                embed_lookup(wf, pt->ids[i], embed_batch + (size_t)i * cfg.hidden_dim);
            }
            double embed_ms = now_ms() - t_embed;
            printf("  [prefill] batch embed %d tokens: %.1f ms\n", pt->count, embed_ms);
        }

        // ---- Batch prefill loop ----
        double prefill_only_ms = 0;  // prefill time excluding last token + LM head
        int prefill_token_count = pt->count > 1 ? pt->count - 1 : 0;
        int prefill_was_batched = 0;
        double t_prefill_start = now_ms();

        if (pt->count > 1 && g_prefill_batch > 1 &&
            (effective_prefill_skip_experts() || g_prefill_experts_full_only)) {
            // Batched prefill (see batched_prefill.h)
            // Linear layers: always K=0 (batched, shared expert only)
            // Full attention layers: batched projections + attention, then:
            //   - if experts_full_only: per-token expert I/O at full-attn layers
            //   - if skip_experts: shared expert only (fastest)
            int num_prefill = pt->count - 1;
            pos += batched_prefill_k0(wf, hidden, embed_batch, num_prefill, pos,
                                       kv_caches, layer_states, layer_mmaps, layer_fds, K);
            prefill_only_ms = now_ms() - t_prefill_start;
            prefill_was_batched = 1;

        } else if (pt->count > 1) {
            // ================================================================
            // ORIGINAL PREFILL: one token at a time through all layers
            // ================================================================
            double t_prefill_batch = now_ms();
            double first_tok_ms = 0;
            int per_tok_K = (g_prefill_k >= 0) ? g_prefill_k :
                            (effective_prefill_skip_experts() ? 0 : K);
            printf("[prefill] starting %d tokens | per-token K=%d skip_experts=%d experts_full_only=%d\n",
                   pt->count - 1, per_tok_K, effective_prefill_skip_experts(), g_prefill_experts_full_only);

            for (int token_idx = 0; token_idx < pt->count - 1; token_idx++) { @autoreleasepool {
                double t_tok = now_ms();

                // Load pre-embedded token from batch buffer
                cache_telemetry_note_token();
                memcpy(hidden, embed_batch + (size_t)token_idx * cfg.hidden_dim,
                       cfg.hidden_dim * sizeof(float));

                int prefill_K = (g_prefill_k >= 0) ? g_prefill_k :
                                (effective_prefill_skip_experts() ? 0 : K);
                for (int layer = 0; layer < cfg.num_layers; layer++) {
                    int is_full = cfg.is_full_attn[layer];
                    int layer_K = prefill_K;
                    if (g_prefill_experts_full_only) {
                        layer_K = is_full ? K : 0;  // full K at full-attn layers, K=0 at linear
                    }
                    fused_layer_forward(wf, layer, hidden,
                                        is_full ? kv_caches[layer] : NULL,
                                        is_full ? NULL : layer_states[layer],
                                        pos,
                                        layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                        layer_K, layer_fds[layer]);
                }

                discard_deferred_experts();
                pos++;

                if (token_idx == 0) {
                    first_tok_ms = now_ms() - t_tok;
                }
            } /* @autoreleasepool */ }

            double prefill_batch_ms = now_ms() - t_prefill_batch;
            double avg_ms = (pt->count > 2) ?
                (prefill_batch_ms - first_tok_ms) / (pt->count - 2) : first_tok_ms;
            printf("  [prefill] %d/%d tokens: %.0f ms (first: %.0f ms, rest avg: %.0f ms)\n",
                   pt->count - 1, pt->count, prefill_batch_ms, first_tok_ms, avg_ms);
            prefill_only_ms = prefill_batch_ms;
            printf("[prefill done] %d tokens | %.0f ms | %.1f tok/s\n",
                   pt->count - 1, prefill_batch_ms,
                   prefill_batch_ms > 0 ? (pt->count - 1) * 1000.0 / prefill_batch_ms : 0);
        }

        // ---- Last prefill token (or single-token prompt) ----
        // This one needs full completion since we need hidden state for logits.
        {
            cache_telemetry_note_token();
            if (embed_batch) {
                memcpy(hidden, embed_batch + (size_t)(pt->count - 1) * cfg.hidden_dim,
                       cfg.hidden_dim * sizeof(float));
            } else {
                embed_lookup(wf, pt->ids[0], hidden);
            }

            for (int layer = 0; layer < cfg.num_layers; layer++) {
                int is_full = cfg.is_full_attn[layer];
                fused_layer_forward(wf, layer, hidden,
                                    is_full ? kv_caches[layer] : NULL,
                                    is_full ? NULL : layer_states[layer],
                                    pos,
                                    layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                    K, layer_fds[layer]);
            }
            // Full completion — need hidden state for final norm + lm_head
            complete_deferred_experts();
            pos++;
        }

        if (embed_batch) { free(embed_batch); embed_batch = NULL; }

        // ---- Final norm ----
        if (final_norm_w) {
            float *normed = malloc(cfg.hidden_dim * sizeof(float));
            cpu_rms_norm(hidden, final_norm_w, normed, cfg.hidden_dim, cfg.rms_norm_eps);
            memcpy(hidden, normed, cfg.hidden_dim * sizeof(float));
            free(normed);
        }

        // ---- LM head ----
        double t_lm = now_ms();
        lm_head_forward(wf, hidden, logits);
        double lm_ms = now_ms() - t_lm;

        // ---- Sample first token ----
        int next_token = cpu_argmax(logits, cfg.vocab_size);
        double ttft_ms = now_ms() - t0;

        // Debug: show top-5 logits for first token
        {
            // Find top 5 manually
            int top5[5] = {0,0,0,0,0};
            float topv[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
            for (int i = 0; i < cfg.vocab_size; i++) {
                int min_k = 0;
                for (int k = 1; k < 5; k++) if (topv[k] < topv[min_k]) min_k = k;
                if (logits[i] > topv[min_k]) { topv[min_k] = logits[i]; top5[min_k] = i; }
            }
            fprintf(stderr, "[debug] Top 5 logits (next_token=%d):\n", next_token);
            for (int i = 0; i < 5; i++) {
                fprintf(stderr, "  token %d (\"%s\") logit=%.4f\n",
                        top5[i], decode_token(vocab, top5[i]), topv[i]);
            }
            fprintf(stderr, "[debug] hidden rms after final_norm=%.4f, logits rms=%.4f\n",
                    vec_rms(hidden, cfg.hidden_dim), vec_rms(logits, cfg.vocab_size));
        }
        printf("[ttft] %.0f ms (prefill %d tokens + lm_head %.0f ms)\n",
               ttft_ms, pt->count, lm_ms);

        printf("\n--- Output ---\n");
        printf("%s", decode_token(vocab, next_token));
        fflush(stdout);

        int total_generated = 1;
        int in_think = (next_token == cfg.think_start_token) ? 1 : 0;
        int think_tokens = 0;

        // ---- Auto-regressive generation ----
        if (g_timing_enabled) timing_reset();
        if (g_pred_enabled) {
            g_pred_generating = 1;  // enable prediction storage/use during generation
            g_pred_valid = 0;       // reset — first gen token builds predictions
        }
        for (int gen = 1; gen < max_tokens; gen++) { @autoreleasepool {
            double t_gen_start = now_ms();

            // Check EOS
            if (next_token == cfg.eos_token_ids[0] || next_token == cfg.eos_token_ids[1]) {
                fprintf(stderr, "\n[eos] Token %d at position %d\n", next_token, gen);
                break;
            }

            // Think budget enforcement
            if (next_token == cfg.think_start_token) in_think = 1;
            if (next_token == cfg.think_end_token) in_think = 0;
            if (in_think) think_tokens++;

            // Embed the just-generated token (next iteration)
            cache_telemetry_note_token();
            embed_lookup(wf, next_token, hidden);

            // Run 40 layers (fused: 1+K cmd buffers per layer)
            for (int layer = 0; layer < cfg.num_layers; layer++) {
                int is_full = cfg.is_full_attn[layer];
                fused_layer_forward(wf, layer, hidden,
                                    is_full ? kv_caches[layer] : NULL,
                                    is_full ? NULL : layer_states[layer],
                                    pos,
                                    layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                    K, layer_fds[layer]);
            }
            // Complete last layer's deferred GPU experts before final norm
            complete_deferred_experts();
            pos++;

            // Final norm
            if (final_norm_w) {
                float *normed = malloc(cfg.hidden_dim * sizeof(float));
                cpu_rms_norm(hidden, final_norm_w, normed, cfg.hidden_dim, cfg.rms_norm_eps);
                memcpy(hidden, normed, cfg.hidden_dim * sizeof(float));
                free(normed);
            }

            // LM head
            lm_head_forward(wf, hidden, logits);

            // Greedy sample
            next_token = cpu_argmax(logits, cfg.vocab_size);

            // Think budget: force end thinking if over budget
            if (in_think && g_think_budget > 0 && think_tokens >= g_think_budget) {
                next_token = cfg.think_end_token;
                in_think = 0;
            }
            total_generated++;

            // Print decoded token
            printf("%s", decode_token(vocab, next_token));
            fflush(stdout);

            double t_gen_end = now_ms();
            double tok_time = t_gen_end - t_gen_start;

            // Print progress to stderr
            fprintf(stderr, "  [gen %d/%d] token_id=%d (%.0f ms, %.2f tok/s)\n",
                    gen, max_tokens, next_token, tok_time, 1000.0 / tok_time);
        } /* @autoreleasepool */ }

        if (g_timing_enabled) timing_print();
        if (!g_stream_mode) {
            printf("\n\n--- Statistics ---\n");
            double total_time = now_ms() - t0;
            printf("Total time:     %.1f s\n", total_time / 1000.0);
            if (prefill_token_count > 0 && prefill_only_ms > 0) {
                printf("TTFT:           %.0f ms (prefill: %d tokens, %.1f tok/s%s)\n",
                       ttft_ms, prefill_token_count,
                       prefill_token_count * 1000.0 / prefill_only_ms,
                       prefill_was_batched ? ", batched" : "");
            } else {
                printf("TTFT:           %.0f ms\n", ttft_ms);
            }
            printf("Tokens:         %d generated\n", total_generated);
            if (total_generated > 1) {
                double gen_time = total_time - ttft_ms;
                printf("Generation:     %.1f s (%.2f tok/s)\n",
                       gen_time / 1000.0, (total_generated - 1) * 1000.0 / gen_time);
            }
            printf("Config:         K=%d experts, %d layers\n", K, cfg.num_layers);
            if (g_expert_cache) {
                uint64_t total = g_expert_cache->hits + g_expert_cache->misses;
                printf("Expert cache:   %llu hits, %llu misses (%.1f%% hit rate), %d/%d entries used\n",
                       g_expert_cache->hits, g_expert_cache->misses,
                       total > 0 ? 100.0 * g_expert_cache->hits / total : 0.0,
                       g_expert_cache->num_entries, g_expert_cache->max_entries);
                cache_telemetry_print(g_expert_cache->hits, g_expert_cache->misses);
            } else if (g_malloc_cache) {
                uint64_t total = g_malloc_cache->hits + g_malloc_cache->misses;
                printf("Expert cache:   malloc %llu hits, %llu misses (%.1f%% hit rate), %d/%d entries used\n",
                       g_malloc_cache->hits, g_malloc_cache->misses,
                       total > 0 ? 100.0 * g_malloc_cache->hits / total : 0.0,
                       g_malloc_cache->num_entries, g_malloc_cache->max_entries);
                cache_telemetry_print(g_malloc_cache->hits, g_malloc_cache->misses);
            }

            if (g_spec_route_attempts > 0) {
                printf("Spec routing:   %llu attempts, %llu preloads, %llu hits (%.1f%% prediction accuracy)\n",
                       g_spec_route_attempts, g_spec_route_preloads, g_spec_route_hits,
                       g_spec_route_attempts > 0
                           ? 100.0 * g_spec_route_hits / g_spec_route_attempts : 0.0);
            }

            if (g_freq_tracking) freq_print_analysis(K);
        }

        if (g_freq_tracking) freq_print_analysis(K);
        if (g_routing_log) {
            fclose(g_routing_log);
            fprintf(stderr, "[routing] Logged %d samples to routing data file\n",
                    g_routing_log_samples);
            g_routing_log = NULL;
        }

        // ---- Cleanup ----
        io_pool_shutdown();
        if (g_malloc_cache) {
            malloc_cache_free(g_malloc_cache);
            g_malloc_cache = NULL;
        }
        if (g_expert_cache) {
            expert_cache_free(g_expert_cache);
            g_expert_cache = NULL;
        }
        for (int i = 0; i < cfg.num_layers; i++) {
            if (kv_caches[i]) kv_cache_free(kv_caches[i]);
            if (layer_states[i]) linear_attn_state_free(layer_states[i]);
            if (layer_mmaps[i] != MAP_FAILED) munmap(layer_mmaps[i], layer_mmap_sizes[i]);
            if (layer_fds[i] >= 0) close(layer_fds[i]);
            if (layer_fds_cold[i] >= 0) close(layer_fds_cold[i]);
        }
        free(layer_states);
        free(kv_caches);
        free(hidden);
        free(logits);

        return 0;
    }
}
#endif // INFER_LIB_MODE

// ============================================================================
// Inference API wrappers (used by server.m when linked as library)
// ============================================================================

// ---- Optimization flag setters (call BEFORE infer_init) ----
void infer_set_fused_expert(int enabled)    { g_fused_expert_enabled = enabled; }
void infer_set_cmd_merge(int enabled)       { g_cmd_merge_enabled = enabled; }
void infer_set_fp8_kv(int enabled)          { g_use_fp8_kv = enabled; }
void infer_set_fused_attention(int enabled) { g_fused_attention_enabled = enabled; }
void infer_set_fp16_accum(int enabled)      { g_use_fp16_accum = enabled; }
void infer_set_expert_prefetch(int enabled) { g_expert_prefetch_enabled = enabled; }
void infer_set_nax(int enabled)            { g_nax_disabled = !enabled; }

void infer_set_prefill_batch(int batch_size) {
    if (batch_size < 1) batch_size = 1;
    if (batch_size > MAX_PFB) batch_size = MAX_PFB;
    g_prefill_batch = batch_size;
}

void infer_set_prefill_skip_experts(int enabled) {
    g_prefill_skip_experts = enabled;
}

void infer_set_prefill_experts_full_only(int enabled) {
    g_prefill_experts_full_only = enabled;
}

void infer_request_prefill_abort(void) {
    atomic_store(&g_prefill_abort, 1);
}

void infer_clear_prefill_abort(void) {
    atomic_store(&g_prefill_abort, 0);
}

int infer_prefill_was_aborted(void) {
    return atomic_load(&g_prefill_abort);
}

int infer_prefill(InferContext *ctx, const uint32_t *token_ids, int num_tokens, int pos_start) {
    if (num_tokens <= 1) return 0;
    int num_prefill = num_tokens - 1;

    // Embed all prefill tokens into a batch buffer
    float *embed_batch = malloc((size_t)num_prefill * cfg.hidden_dim * sizeof(float));
    if (!embed_batch) {
        fprintf(stderr, "[prefill] ERROR: failed to allocate embed_batch (%d tokens)\n", num_prefill);
        return 0;
    }
    for (int i = 0; i < num_prefill; i++) {
        embed_lookup(ctx->wf, token_ids[i], embed_batch + (size_t)i * cfg.hidden_dim);
    }

    int result;
    if (g_prefill_batch > 1 &&
        (effective_prefill_skip_experts() || g_prefill_experts_full_only)) {
        // Batched prefill path
        result = batched_prefill_k0(ctx->wf, ctx->hidden, embed_batch,
                                     num_prefill, pos_start,
                                     ctx->kv_caches, ctx->layer_states,
                                     ctx->layer_mmaps, ctx->layer_fds, ctx->K);
    } else {
        // Per-token prefill fallback
        int prefill_K = (effective_prefill_skip_experts() || g_prefill_experts_full_only) ? 0 : ctx->K;
        int i;
        for (i = 0; i < num_prefill; i++) {
            if (atomic_load(&g_prefill_abort)) {
                fprintf(stderr, "[prefill] aborted at token %d/%d\n", i, num_prefill);
                break;
            }
            @autoreleasepool {
            cache_telemetry_note_token();
            memcpy(ctx->hidden, embed_batch + (size_t)i * cfg.hidden_dim,
                   cfg.hidden_dim * sizeof(float));
            for (int layer = 0; layer < cfg.num_layers; layer++) {
                int is_full = cfg.is_full_attn[layer];
                int layer_K = prefill_K;
                if (g_prefill_experts_full_only) {
                    layer_K = is_full ? ctx->K : 0;
                }
                fused_layer_forward(ctx->wf, layer, ctx->hidden,
                    is_full ? ctx->kv_caches[layer] : NULL,
                    is_full ? NULL : (LinearAttnState *)ctx->layer_states[layer],
                    pos_start + i,
                    ctx->layer_mmaps[layer] != MAP_FAILED ? ctx->layer_mmaps[layer] : NULL,
                    layer_K, ctx->layer_fds[layer]);
            }
            discard_deferred_experts();
            } // @autoreleasepool
        }
        result = i;
    }

    free(embed_batch);
    return result;
}

void infer_load_model_config(const char *model_dir) {
    load_model_config(model_dir ? model_dir : "");
}

InferContext *infer_init(const char *model_path,
                         const char *weights_path_arg,
                         const char *manifest_path_arg,
                         const char *vocab_path_arg,
                         int K, int use_tiered, int use_2bit) {
    // Load model configuration
    infer_load_model_config(model_path);

    // Resolve default paths
    char default_weights[1024], default_manifest[1024], default_vocab[1024];
    const char *weights_path = weights_path_arg;
    const char *manifest_path = manifest_path_arg;
    const char *vocab_path = vocab_path_arg;

    if (!weights_path) {
        if (model_path) snprintf(default_weights, sizeof(default_weights), "%s/model_weights.bin", model_path);
        if (!model_path || access(default_weights, R_OK) != 0) {
            snprintf(default_weights, sizeof(default_weights), "model_weights.bin");
        }
        weights_path = default_weights;
    }
    if (!manifest_path) {
        if (model_path) snprintf(default_manifest, sizeof(default_manifest), "%s/model_weights.json", model_path);
        if (!model_path || access(default_manifest, R_OK) != 0) {
            snprintf(default_manifest, sizeof(default_manifest), "model_weights.json");
        }
        manifest_path = default_manifest;
    }
    if (!vocab_path) {
        if (model_path) snprintf(default_vocab, sizeof(default_vocab), "%s/vocab.bin", model_path);
        if (!model_path || access(default_vocab, R_OK) != 0) {
            snprintf(default_vocab, sizeof(default_vocab), "vocab.bin");
        }
        vocab_path = default_vocab;
    }

    // Allocate tracking arrays + deferred expert state buffer
    extern void alloc_tracking_arrays(void);
    alloc_tracking_arrays();
    g_deferred.h_mid = calloc(cfg.hidden_dim, sizeof(float));

    // Initialize Metal
    extern MetalCtx *g_metal;
    g_metal = metal_setup();
    if (!g_metal) {
        fprintf(stderr, "WARNING: Metal init failed, falling back to CPU\n");
    }

    // Initialize I/O thread pool
    io_pool_init();

    // Handle tiered/2bit modes
    extern int g_use_tiered, g_use_2bit;
    g_use_tiered = use_tiered;
    g_use_2bit = use_2bit;

    // Load weights
    WeightFile *wf = open_weights(weights_path, manifest_path);
    if (!wf) {
        fprintf(stderr, "ERROR: Failed to load weights\n");
        return NULL;
    }
    if (g_metal) {
        metal_set_weights(g_metal, wf->data, wf->size);
    }

    // Load vocabulary
    Vocabulary *vocab = load_vocab(vocab_path);
    if (!vocab) {
        fprintf(stderr, "ERROR: Failed to load vocabulary\n");
        return NULL;
    }

    // Auto-detect tiered experts
    extern int load_tiered_manifest(const char *model_path);
    if (!g_use_2bit && !g_use_tiered && model_path) {
        char probe[1024];
        snprintf(probe, sizeof(probe), "%s/packed_experts_tiered/tiered_manifest.json", model_path);
        if (access(probe, F_OK) == 0) {
            if (load_tiered_manifest(model_path)) {
                g_use_tiered = 1;
            }
        }
    }
    if (g_use_tiered) {
        extern TieredExpertInfo *g_tiered_manifest;
        if (!g_tiered_manifest) {
            if (!load_tiered_manifest(model_path)) {
                fprintf(stderr, "WARNING: --tiered but no tiered_manifest.json found\n");
                g_use_tiered = 0;
            }
        }
    }

    // Open packed expert files
    int *layer_fds = calloc(cfg.num_layers, sizeof(int));
    void **layer_mmaps = calloc(cfg.num_layers, sizeof(void *));
    int expert_layers = 0;

    for (int i = 0; i < cfg.num_layers; i++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s/layer_%02d.bin", model_path,
                 g_use_tiered ? "packed_experts_tiered" :
                 g_use_2bit ? "packed_experts_2bit" : "packed_experts", i);
        layer_fds[i] = open(path, O_RDONLY);
        layer_mmaps[i] = MAP_FAILED;
        if (layer_fds[i] >= 0) {
            expert_layers++;
            fcntl(layer_fds[i], F_RDAHEAD, 0);
            struct stat st;
            if (fstat(layer_fds[i], &st) == 0 && st.st_size > 0) {
                layer_mmaps[i] = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, layer_fds[i], 0);
                if (layer_mmaps[i] == MAP_FAILED) layer_mmaps[i] = MAP_FAILED;
            }
        }
    }
    printf("[experts] %d/%d packed layer files available\n", expert_layers, cfg.num_layers);

    // Cold fd setup
    extern int *g_layer_fds_cold;
    int *layer_fds_cold = calloc(cfg.num_layers, sizeof(int));
    for (int i = 0; i < cfg.num_layers; i++) layer_fds_cold[i] = -1;
    g_layer_fds_cold = layer_fds_cold;

    // Set global layer fds for cross-layer expert prefetch
    if (g_expert_prefetch_enabled) {
        g_expert_prefetch_layer_fds = layer_fds;
    }

    // Warm page cache
    for (int i = 0; i < cfg.num_layers; i++) {
        if (layer_fds[i] >= 0) {
            char dummy[4096];
            pread(layer_fds[i], dummy, sizeof(dummy), 0);
        }
    }

    // Allocate per-layer state
    void **layer_states = calloc(cfg.num_layers, sizeof(void *));
    KVCache **kv_caches = (KVCache **)calloc(cfg.num_layers, sizeof(KVCache *));

    for (int i = 0; i < cfg.num_layers; i++) {
        int is_full = cfg.is_full_attn[i];
        if (is_full) {
            kv_caches[i] = kv_cache_new();
        } else {
            layer_states[i] = linear_attn_state_new();
        }
    }

    // Allocate working buffers
    float *hidden = calloc(cfg.hidden_dim, sizeof(float));
    float *logits = calloc(cfg.vocab_size, sizeof(float));
    uint16_t *final_norm_w = get_tensor_ptr(wf, "model.norm.weight");

    // Build context
    InferContext *ctx = calloc(1, sizeof(InferContext));
    ctx->wf = wf;
    ctx->vocab = vocab;
    ctx->layer_states = layer_states;
    ctx->kv_caches = kv_caches;
    ctx->layer_mmaps = layer_mmaps;
    ctx->layer_fds = layer_fds;
    ctx->hidden = hidden;
    ctx->logits = logits;
    ctx->final_norm_w = final_norm_w;
    ctx->K = K;

    printf("[server] Inference context initialized (K=%d, %s)\n", K,
           g_use_tiered ? "tiered" : g_use_2bit ? "2-bit" : "4-bit");
    return ctx;
}

void infer_shutdown(InferContext *ctx) {
    if (!ctx) return;
    free(ctx->hidden);
    free(ctx->logits);
    for (int i = 0; i < cfg.num_layers; i++) {
        if (ctx->layer_fds[i] >= 0) close(ctx->layer_fds[i]);
    }
    free(ctx->layer_states);
    free(ctx->kv_caches);
    free(ctx->layer_mmaps);
    free(ctx->layer_fds);
    free(ctx);
}

PromptTokens *infer_encode_text(const char *text) {
    return encode_prompt_text_to_tokens(text);
}

void infer_free_tokens(PromptTokens *pt) {
    if (pt) {
        free(pt->ids);
        free(pt);
    }
}

const char *infer_decode_token(InferContext *ctx, int token_id) {
    return decode_token(ctx->vocab, token_id);
}

void infer_embed_token(InferContext *ctx, int token_id) {
    embed_lookup(ctx->wf, token_id, ctx->hidden);
}

void infer_forward_layer(InferContext *ctx, int layer, int pos) {
    int is_full = cfg.is_full_attn[layer];
    fused_layer_forward(ctx->wf, layer, ctx->hidden,
        is_full ? ctx->kv_caches[layer] : NULL,
        is_full ? NULL : (LinearAttnState *)ctx->layer_states[layer],
        pos,
        ctx->layer_mmaps[layer] != MAP_FAILED ? ctx->layer_mmaps[layer] : NULL,
        ctx->K, ctx->layer_fds[layer]);
}

void infer_complete_deferred(void) {
    complete_deferred_experts();
}

void infer_discard_deferred(void) {
    discard_deferred_experts();
}

// No-wait variant: same as infer_discard_deferred() but semantically marks
// that we don't need the result. Currently identical implementation because
// GPU buffer safety requires waiting, but kept as a separate entry point
// for future optimization (if GPU queue serialization makes the wait redundant).
void infer_discard_deferred_nowait(void) {
    discard_deferred_experts();
}

void infer_final_norm(InferContext *ctx) {
    if (ctx->final_norm_w) {
        float *normed = malloc(cfg.hidden_dim * sizeof(float));
        cpu_rms_norm(ctx->hidden, ctx->final_norm_w, normed, cfg.hidden_dim, cfg.rms_norm_eps);
        memcpy(ctx->hidden, normed, cfg.hidden_dim * sizeof(float));
        free(normed);
    }
}

void infer_lm_head(InferContext *ctx) {
    lm_head_forward(ctx->wf, ctx->hidden, ctx->logits);
}

int infer_argmax(InferContext *ctx) {
    return cpu_argmax(ctx->logits, cfg.vocab_size);
}

float *infer_get_logits(InferContext *ctx) {
    return ctx->logits;
}

int infer_is_eos(int token_id) {
    for (int i = 0; i < cfg.num_eos_tokens; i++) {
        if (token_id == cfg.eos_token_ids[i]) return 1;
    }
    return 0;
}

void infer_reset_state(InferContext *ctx) {
    reset_delta_net_state();
    size_t kv_dim = cfg.num_kv_heads * cfg.head_dim;
    for (int i = 0; i < cfg.num_layers; i++) {
        if (ctx->kv_caches[i]) {
            // Only zero the portion actually used (up to len), not the full max_seq_len.
            // calloc already zeroed the rest. This avoids touching 5+ GB of virtual memory
            // and forcing physical page allocation on startup.
            int used = ctx->kv_caches[i]->len;
            if (used > 0) {
                memset(ctx->kv_caches[i]->k_cache, 0, (size_t)used * kv_dim * sizeof(float));
                memset(ctx->kv_caches[i]->v_cache, 0, (size_t)used * kv_dim * sizeof(float));
            }
            ctx->kv_caches[i]->len = 0;
        }
        if (ctx->layer_states[i]) {
            LinearAttnState *s = (LinearAttnState *)ctx->layer_states[i];
            memset(s->conv_state, 0, (cfg.conv_kernel_size - 1) * cfg.linear_conv_dim * sizeof(float));
            memset(s->ssm_state, 0, cfg.linear_num_v_heads * cfg.linear_value_dim * cfg.linear_key_dim * sizeof(float));
        }
    }
}

void infer_sync_delta_state(InferContext *ctx) {
    sync_cpu_to_gpu_delta_state_serve(ctx->layer_states);
}

InferStateSnapshot *infer_snapshot_state(InferContext *ctx, int pos) {
    InferStateSnapshot *snap = calloc(1, sizeof(InferStateSnapshot));
    snap->pos = pos;

    size_t kv_dim = cfg.num_kv_heads * cfg.head_dim;
    size_t conv_state_size = (cfg.conv_kernel_size - 1) * cfg.linear_conv_dim * sizeof(float);
    size_t ssm_state_size = cfg.linear_num_v_heads * cfg.linear_value_dim * cfg.linear_key_dim * sizeof(float);

    snap->kv_k_snapshots = calloc(cfg.num_layers, sizeof(float *));
    snap->kv_v_snapshots = calloc(cfg.num_layers, sizeof(float *));
    snap->kv_lens = calloc(cfg.num_layers, sizeof(int));
    snap->la_conv_snapshots = calloc(cfg.num_layers, sizeof(float *));
    snap->la_ssm_snapshots = calloc(cfg.num_layers, sizeof(float *));

    for (int i = 0; i < cfg.num_layers; i++) {
        if (ctx->kv_caches[i]) {
            size_t sz = pos * kv_dim * sizeof(float);
            snap->kv_k_snapshots[i] = malloc(sz);
            snap->kv_v_snapshots[i] = malloc(sz);
            memcpy(snap->kv_k_snapshots[i], ctx->kv_caches[i]->k_cache, sz);
            memcpy(snap->kv_v_snapshots[i], ctx->kv_caches[i]->v_cache, sz);
            snap->kv_lens[i] = ctx->kv_caches[i]->len;
        }
        if (ctx->layer_states[i]) {
            LinearAttnState *s = (LinearAttnState *)ctx->layer_states[i];
            snap->la_conv_snapshots[i] = malloc(conv_state_size);
            snap->la_ssm_snapshots[i] = malloc(ssm_state_size);
            memcpy(snap->la_conv_snapshots[i], s->conv_state, conv_state_size);
            memcpy(snap->la_ssm_snapshots[i], s->ssm_state, ssm_state_size);
        }
    }

    // GPU delta-net snapshots
    extern MetalCtx *g_metal;
    snap->gpu_delta_snapshots = calloc(cfg.num_linear_layers, sizeof(void *));
    snap->gpu_conv_snapshots = calloc(cfg.num_linear_layers, sizeof(void *));
    if (g_metal && g_metal->delta_net_step) {
        for (int i = 0; i < cfg.num_linear_layers; i++) {
            if (g_metal->buf_delta_state[i]) {
                size_t sz = (size_t)cfg.linear_num_v_heads * cfg.linear_value_dim * cfg.linear_key_dim * sizeof(float);
                snap->gpu_delta_snapshots[i] = malloc(sz);
                memcpy(snap->gpu_delta_snapshots[i], [g_metal->buf_delta_state[i] contents], sz);
            }
            if (g_metal->buf_conv_state[i]) {
                size_t sz = (cfg.conv_kernel_size - 1) * (size_t)cfg.linear_conv_dim * sizeof(float);
                snap->gpu_conv_snapshots[i] = malloc(sz);
                memcpy(snap->gpu_conv_snapshots[i], [g_metal->buf_conv_state[i] contents], sz);
            }
        }
    }

    return snap;
}

void infer_restore_state(InferContext *ctx, InferStateSnapshot *snap) {
    if (!snap) return;

    size_t kv_dim = cfg.num_kv_heads * cfg.head_dim;
    size_t conv_state_size = (cfg.conv_kernel_size - 1) * cfg.linear_conv_dim * sizeof(float);
    size_t ssm_state_size = cfg.linear_num_v_heads * cfg.linear_value_dim * cfg.linear_key_dim * sizeof(float);

    extern MetalCtx *g_metal;
    for (int i = 0; i < cfg.num_layers; i++) {
        if (ctx->kv_caches[i] && snap->kv_k_snapshots[i]) {
            size_t sz = snap->pos * kv_dim * sizeof(float);
            memcpy(ctx->kv_caches[i]->k_cache, snap->kv_k_snapshots[i], sz);
            memcpy(ctx->kv_caches[i]->v_cache, snap->kv_v_snapshots[i], sz);
            ctx->kv_caches[i]->len = snap->kv_lens[i];
            // Restore GPU KV mirror
            if (g_metal) {
                int fa_idx = cfg.full_attn_index[i];
                if (fa_idx >= 0 && fa_idx < cfg.num_full_attn_layers) {
                    memcpy([g_metal->buf_kv_k[fa_idx] contents], snap->kv_k_snapshots[i], sz);
                    memcpy([g_metal->buf_kv_v[fa_idx] contents], snap->kv_v_snapshots[i], sz);
                }
            }
        } else if (ctx->kv_caches[i]) {
            ctx->kv_caches[i]->len = 0;
        }
        if (ctx->layer_states[i] && snap->la_conv_snapshots[i]) {
            LinearAttnState *s = (LinearAttnState *)ctx->layer_states[i];
            memcpy(s->conv_state, snap->la_conv_snapshots[i], conv_state_size);
            memcpy(s->ssm_state, snap->la_ssm_snapshots[i], ssm_state_size);
        } else if (ctx->layer_states[i]) {
            LinearAttnState *s = (LinearAttnState *)ctx->layer_states[i];
            memset(s->conv_state, 0, conv_state_size);
            memset(s->ssm_state, 0, ssm_state_size);
        }
    }
    // Restore GPU delta-net state
    if (g_metal && g_metal->delta_net_step) {
        for (int i = 0; i < cfg.num_linear_layers; i++) {
            if (snap->gpu_delta_snapshots[i] && g_metal->buf_delta_state[i])
                memcpy([g_metal->buf_delta_state[i] contents], snap->gpu_delta_snapshots[i],
                       (size_t)cfg.linear_num_v_heads * cfg.linear_value_dim * cfg.linear_key_dim * sizeof(float));
            if (snap->gpu_conv_snapshots[i] && g_metal->buf_conv_state[i])
                memcpy([g_metal->buf_conv_state[i] contents], snap->gpu_conv_snapshots[i],
                       (cfg.conv_kernel_size - 1) * (size_t)cfg.linear_conv_dim * sizeof(float));
        }
    } else {
        reset_delta_net_state();
    }
}

void infer_free_snapshot(InferStateSnapshot *snap) {
    if (!snap) return;
    for (int i = 0; i < cfg.num_layers; i++) {
        free(snap->kv_k_snapshots[i]);
        free(snap->kv_v_snapshots[i]);
        free(snap->la_conv_snapshots[i]);
        free(snap->la_ssm_snapshots[i]);
    }
    for (int i = 0; i < cfg.num_linear_layers; i++) {
        free(snap->gpu_delta_snapshots[i]);
        free(snap->gpu_conv_snapshots[i]);
    }
    free(snap->kv_k_snapshots);
    free(snap->kv_v_snapshots);
    free(snap->kv_lens);
    free(snap->la_conv_snapshots);
    free(snap->la_ssm_snapshots);
    free(snap->gpu_delta_snapshots);
    free(snap->gpu_conv_snapshots);
    free(snap);
}

#endif // CHAT_MODE
