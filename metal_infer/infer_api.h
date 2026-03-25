/*
 * infer_api.h — Public API for the Flash-MoE inference engine
 *
 * Exposes the minimum interface needed by server.m to drive inference
 * without touching Metal internals directly.
 *
 * Link with: infer.o (compiled from infer.m with -DINFER_LIB_MODE)
 */

#ifndef INFER_API_H
#define INFER_API_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// ============================================================================
// Types
// ============================================================================

// Runtime model configuration — populated from HuggingFace config.json
typedef struct {
    // Core architecture
    int hidden_dim;
    int num_layers;
    int num_attn_heads;
    int num_kv_heads;
    int head_dim;
    int vocab_size;
    float rms_norm_eps;

    // Model type: 0=Qwen3.5, 1=MiniMax
    int model_type;
    char moe_prefix[32];          // "mlp" (Qwen) or "block_sparse_moe" (MiniMax)
    int has_attn_gate;            // 1=Qwen (Q proj includes sigmoid gate), 0=MiniMax
    int scoring_func;             // 0=softmax (Qwen), 1=sigmoid (MiniMax)
    int qk_norm_per_layer;        // 0=per-head (Qwen), 1=per-layer (MiniMax)

    // MoE
    int num_experts;
    int num_experts_per_tok;
    int moe_intermediate;
    int shared_intermediate;
    int group_size;
    int bits;
    int gate_bits;                // routing gate quantization bits (may differ from expert bits)
    int gate_group_size;          // routing gate group size

    // Linear attention (GatedDeltaNet)
    int linear_num_v_heads;
    int linear_num_k_heads;
    int linear_key_dim;
    int linear_value_dim;
    int conv_kernel_size;

    // Full attention
    float rope_theta;
    float partial_rotary;

    // Layer type map
    int num_full_attn_layers;
    int num_linear_layers;
    bool *is_full_attn;
    int *full_attn_index;
    int *linear_index;

    // Expert byte offsets (4-bit)
    size_t expert_size_4bit;
    size_t gate_w_off_4, gate_s_off_4, gate_b_off_4;
    size_t up_w_off_4, up_s_off_4, up_b_off_4;
    size_t down_w_off_4, down_s_off_4, down_b_off_4;

    // Expert byte offsets (2-bit)
    size_t expert_size_2bit;
    size_t gate_w_off_2, gate_s_off_2, gate_b_off_2;
    size_t up_w_off_2, up_s_off_2, up_b_off_2;
    size_t down_w_off_2, down_s_off_2, down_b_off_2;

    // Derived dimensions
    int linear_total_key;
    int linear_total_value;
    int linear_conv_dim;
    int rotary_dim;

    // Special tokens
    int eos_token_ids[8];
    int num_eos_tokens;
    int think_start_token;
    int think_end_token;

    // Context limits
    int max_seq_len;
    int gpu_kv_seq;

    // Model path (resolved)
    char model_path[1024];
} ModelConfig;

// Prompt tokens
typedef struct {
    uint32_t *ids;
    int count;
} PromptTokens;

// Opaque types — server.m holds pointers but never touches internals
typedef struct WeightFile_s WeightFile;
typedef struct Vocabulary_s Vocabulary;
typedef struct KVCache_s KVCache;
typedef struct LinearAttnState_s LinearAttnState;

// Inference context — bundles all state needed for generation
typedef struct {
    WeightFile   *wf;
    Vocabulary   *vocab;
    void        **layer_states;    // [num_layers] LinearAttnState* or NULL
    KVCache     **kv_caches;       // [num_layers] KVCache* or NULL
    void        **layer_mmaps;     // [num_layers] mmap pointers
    int          *layer_fds;       // [num_layers] file descriptors
    float        *hidden;          // [hidden_dim] working buffer
    float        *logits;          // [vocab_size] output logits
    uint16_t     *final_norm_w;    // final RMS norm weights (bf16)
    int           K;               // active experts per layer
} InferContext;

// ============================================================================
// Global config (populated by infer_load_model_config)
// ============================================================================

extern ModelConfig cfg;

// ============================================================================
// Initialization
// ============================================================================

// Load model config from HF config.json in model_dir
void infer_load_model_config(const char *model_dir);

// Initialize full inference engine: Metal, I/O pool, weights, vocab, KV caches.
// Returns an InferContext that bundles all state. Caller must free with infer_shutdown().
// Any NULL path parameter uses auto-detection from model_path.
InferContext *infer_init(const char *model_path,
                         const char *weights_path,
                         const char *manifest_path,
                         const char *vocab_path,
                         int K, int use_tiered, int use_2bit);

// Clean up inference context
void infer_shutdown(InferContext *ctx);

// ============================================================================
// Tokenization
// ============================================================================

// Encode raw text to token IDs via BPE tokenizer
PromptTokens *infer_encode_text(const char *text);

// Free prompt tokens
void infer_free_tokens(PromptTokens *pt);

// ============================================================================
// Token decoding
// ============================================================================

// Decode token ID to UTF-8 string (pointer valid until next call)
const char *infer_decode_token(InferContext *ctx, int token_id);

// ============================================================================
// Inference — single token step
// ============================================================================

// Embed a token into the hidden buffer
void infer_embed_token(InferContext *ctx, int token_id);

// Forward hidden through one transformer layer
void infer_forward_layer(InferContext *ctx, int layer, int pos);

// GPU sync: complete deferred expert computation (read back + combine)
void infer_complete_deferred(void);

// GPU sync: discard deferred results (used for intermediate prefill tokens)
void infer_discard_deferred(void);

// GPU sync: discard deferred results without requiring immediate GPU wait
// (currently same as infer_discard_deferred for buffer safety)
void infer_discard_deferred_nowait(void);

// Apply final RMS norm to hidden buffer
void infer_final_norm(InferContext *ctx);

// Project hidden → logits via lm_head
void infer_lm_head(InferContext *ctx);

// Greedy decode: argmax over logits
int infer_argmax(InferContext *ctx);

// Raw argmax over a float array (used by sampling)
int cpu_argmax(const float *x, int dim);

// Get pointer to logits buffer (for custom sampling)
float *infer_get_logits(InferContext *ctx);

// ============================================================================
// Special tokens
// ============================================================================

// Check if token_id is an EOS token
int infer_is_eos(int token_id);

// ============================================================================
// State management
// ============================================================================

// Reset all KV caches and linear attention state to zero
void infer_reset_state(InferContext *ctx);

// Sync CPU linear attention state → GPU buffers
void infer_sync_delta_state(InferContext *ctx);

// ============================================================================
// State snapshots (for system prompt caching)
// ============================================================================

typedef struct {
    // KV cache snapshots
    float **kv_k_snapshots;    // [num_layers] or NULL
    float **kv_v_snapshots;    // [num_layers] or NULL
    int   *kv_lens;            // [num_layers]
    // Linear attention snapshots
    float **la_conv_snapshots; // [num_layers] or NULL
    float **la_ssm_snapshots;  // [num_layers] or NULL
    // GPU delta-net snapshots
    void  **gpu_delta_snapshots; // [num_linear_layers] or NULL
    void  **gpu_conv_snapshots;  // [num_linear_layers] or NULL
    int     pos;               // position after snapshot
} InferStateSnapshot;

// Take a snapshot of the current inference state
InferStateSnapshot *infer_snapshot_state(InferContext *ctx, int pos);

// Restore inference state from a snapshot
void infer_restore_state(InferContext *ctx, InferStateSnapshot *snap);

// Free a snapshot
void infer_free_snapshot(InferStateSnapshot *snap);

// ============================================================================
// Optimization flags (call BEFORE infer_init)
// ============================================================================

// Fused gate+up+SwiGLU expert kernel (default ON for 4-bit)
void infer_set_fused_expert(int enabled);

// CMD1+CMD2 merge for linear attention layers (default ON)
void infer_set_cmd_merge(int enabled);

// FP8 E4M3 KV cache — 4x memory reduction (default OFF)
void infer_set_fp8_kv(int enabled);

// Fused online softmax attention (default OFF, experimental)
void infer_set_fused_attention(int enabled);

// FP16 accumulation in dequant kernels (default OFF, experimental)
void infer_set_fp16_accum(int enabled);

// Cross-layer expert prefetch (default OFF)
void infer_set_expert_prefetch(int enabled);

// NAX tensor matmul for LM head (Metal 4 / M5+, default OFF — slower for M=1 decode)
void infer_set_nax(int enabled);

// ============================================================================
// Batched prefill (call BEFORE infer_init to configure, then use infer_prefill)
// ============================================================================

// Set prefill batch size (default 1 = no batching, recommended 64-128)
void infer_set_prefill_batch(int batch_size);

// Skip routed experts during prefill (shared expert only, fastest)
void infer_set_prefill_skip_experts(int enabled);

// K=0 for linear layers, full K for full-attn layers (best quality)
void infer_set_prefill_experts_full_only(int enabled);

// Batched prefill: process all intermediate prompt tokens through the model.
// Embeds tokens, runs batched prefill (or per-token fallback), updates pos.
// Does NOT process the last token — caller must handle it for full completion.
// Returns number of tokens prefilled (= num_tokens - 1), or 0 if num_tokens <= 1.
// Returns fewer tokens if aborted early (check infer_prefill_was_aborted()).
int infer_prefill(InferContext *ctx, const uint32_t *token_ids, int num_tokens, int pos_start);

// Prefill abort: signal prefill to stop early (thread-safe, atomic).
// Call infer_request_prefill_abort() from any thread to interrupt an in-progress prefill.
// The prefill loop checks this flag every token (per-token path) or every layer (batched path).
// After prefill returns, check infer_prefill_was_aborted() and call infer_clear_prefill_abort().
void infer_request_prefill_abort(void);
void infer_clear_prefill_abort(void);
int infer_prefill_was_aborted(void);

#endif // INFER_API_H
