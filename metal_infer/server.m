/*
 * server.m — OpenAI-compatible API server for Flash-MoE
 *
 * Provides /v1/chat/completions (streaming + non-streaming),
 * /v1/models, /health endpoints.
 *
 * Features:
 *   - Full OpenAI chat completion request/response format
 *   - SSE streaming with proper chunked deltas
 *   - Temperature, top-p, top-k sampling
 *   - Frequency/presence penalty
 *   - Stop sequence detection across token boundaries
 *   - Tool call detection (<tool_call> XML → OpenAI JSON)
 *   - Request queue with concurrent HTTP accept + serial inference
 *   - System prompt caching (snapshot restore, no full KV memset)
 *   - Usage statistics (prompt_tokens, completion_tokens, total_tokens)
 *
 * Build: make server  (requires infer.o from infer.m)
 * Run:   ./server --model /path/to/model --port 8080
 */

#import <Foundation/Foundation.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <signal.h>
#include <pthread.h>
#include <math.h>
#include <sys/time.h>
#include <getopt.h>
#include <errno.h>
#include <poll.h>

#include "infer_api.h"

// ============================================================================
// Constants
// ============================================================================

#define MAX_REQUEST_SIZE  (4 * 1024 * 1024)
#define MAX_STOP_SEQUENCES 8
#define MAX_STOP_SEQ_LEN   128
#define MAX_MESSAGES       512
#define DEFAULT_PORT       8080
#define REQUEST_QUEUE_SIZE 32
#define MAX_GEN_TOKENS     32768
#define RESPONSE_BUF_SIZE  (512 * 1024)

// ============================================================================
// Utilities
// ============================================================================

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// Check if client disconnected (non-blocking poll for hangup/error)
static int client_disconnected(int fd) {
    struct pollfd pfd = { .fd = fd, .events = 0 };
    if (poll(&pfd, 1, 0) > 0 && (pfd.revents & (POLLHUP | POLLERR))) return 1;
    return 0;
}

static void generate_request_id(char *buf, int bufsize) {
    static uint64_t counter = 0;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    snprintf(buf, bufsize, "chatcmpl-%llu-%ld", (unsigned long long)++counter, tv.tv_sec);
}

static void generate_call_id(char *buf, int bufsize) {
    static uint64_t call_counter = 0;
    snprintf(buf, bufsize, "call_%llu", (unsigned long long)++call_counter);
}

// ============================================================================
// JSON string escaping
// ============================================================================

static int json_escape(const char *src, char *dst, int dst_size) {
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

// ============================================================================
// HTTP helpers
// ============================================================================

static void http_write(int fd, const char *data, size_t len) {
    size_t written = 0;
    while (written < len) {
        ssize_t n = write(fd, data + written, len - written);
        if (n <= 0) break;
        written += n;
    }
}

static void http_write_str(int fd, const char *s) {
    http_write(fd, s, strlen(s));
}

static int read_http_request(int fd, char *buf, int bufsz) {
    int total = 0;
    // Read headers
    while (total < bufsz - 1) {
        ssize_t n = read(fd, buf + total, 1);
        if (n <= 0) return total > 0 ? total : -1;
        total++;
        if (total >= 4 &&
            buf[total-4] == '\r' && buf[total-3] == '\n' &&
            buf[total-2] == '\r' && buf[total-1] == '\n') {
            break;
        }
    }
    buf[total] = '\0';

    // Find Content-Length
    char *cl = strcasestr(buf, "content-length:");
    if (cl) {
        int body_len = atoi(cl + 15);
        if (body_len > 0 && total + body_len < bufsz) {
            int body_read = 0;
            while (body_read < body_len) {
                ssize_t n = read(fd, buf + total + body_read, body_len - body_read);
                if (n <= 0) break;
                body_read += n;
            }
            total += body_read;
            buf[total] = '\0';
        }
    }
    return total;
}

static const char *CORS_HEADERS =
    "Access-Control-Allow-Origin: *\r\n"
    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
    "Access-Control-Allow-Headers: Content-Type, Authorization\r\n";

static void send_json_response(int fd, int status, const char *body) {
    const char *status_text = "OK";
    if (status == 400) status_text = "Bad Request";
    else if (status == 404) status_text = "Not Found";
    else if (status == 500) status_text = "Internal Server Error";
    else if (status == 503) status_text = "Service Unavailable";

    char header[512];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: application/json\r\n"
        "%s"
        "Connection: close\r\n"
        "Content-Length: %zu\r\n"
        "\r\n",
        status, status_text, CORS_HEADERS, strlen(body));
    http_write(fd, header, hlen);
    http_write_str(fd, body);
}

static void send_error(int fd, int status, const char *message, const char *type) {
    char escaped_msg[2048];
    json_escape(message, escaped_msg, sizeof(escaped_msg));
    char body[4096];
    snprintf(body, sizeof(body),
        "{\"error\":{\"message\":\"%s\",\"type\":\"%s\",\"param\":null,\"code\":null}}",
        escaped_msg, type);
    send_json_response(fd, status, body);
}

// ============================================================================
// Completion request parsing
// ============================================================================

typedef struct {
    // Parsed from messages array
    char **msg_roles;           // [num_messages]
    char **msg_contents;        // [num_messages]
    char **msg_tool_call_ids;   // [num_messages] for role="tool"
    int    num_messages;

    // Sampling parameters
    float  temperature;         // 0.0-2.0, default 1.0
    float  top_p;               // 0.0-1.0, default 1.0
    int    top_k;               // 0 = disabled
    int    max_tokens;          // max completion tokens
    int    min_tokens;          // min tokens before EOS/stop allowed (0 = disabled)
    float  frequency_penalty;   // -2.0 to 2.0
    float  presence_penalty;    // -2.0 to 2.0

    // Stop sequences
    char   stop_seqs[MAX_STOP_SEQUENCES][MAX_STOP_SEQ_LEN];
    int    num_stop_seqs;

    // Streaming
    bool   stream;
    bool   include_usage;       // stream_options.include_usage

    // Tools
    char  *tools_json;          // raw JSON string of tools array (for system prompt injection)
    char  *tool_choice;         // "auto", "none", "required"

    // Session-based KV caching
    bool   cache;               // reuse KV state from previous turn
    char   session_id[64];      // client session ID for KV cache continuity

    // Request metadata
    char   model[128];
    char   request_id[64];
    long   created;
} CompletionRequest;

static void free_completion_request(CompletionRequest *req) {
    for (int i = 0; i < req->num_messages; i++) {
        free(req->msg_roles[i]);
        free(req->msg_contents[i]);
        free(req->msg_tool_call_ids[i]);
    }
    free(req->msg_roles);
    free(req->msg_contents);
    free(req->msg_tool_call_ids);
    free(req->tools_json);
    free(req->tool_choice);
}

static int parse_completion_request(const char *body, int body_len, CompletionRequest *req) {
    memset(req, 0, sizeof(*req));

    // Defaults
    req->temperature = 1.0f;
    req->top_p = 1.0f;
    req->top_k = 0;
    req->max_tokens = 4096;
    req->stream = false;
    req->include_usage = false;

    // Parse JSON with NSJSONSerialization
    NSData *data = [NSData dataWithBytes:body length:body_len];
    NSError *error = nil;
    NSDictionary *json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    if (!json || ![json isKindOfClass:[NSDictionary class]]) {
        return -1;
    }

    // Model
    NSString *model = json[@"model"];
    if (model && [model isKindOfClass:[NSString class]]) {
        strlcpy(req->model, [model UTF8String], sizeof(req->model));
    }

    // Messages
    NSArray *messages = json[@"messages"];
    if (!messages || ![messages isKindOfClass:[NSArray class]] || [messages count] == 0) {
        return -1;
    }

    int n = (int)[messages count];
    if (n > MAX_MESSAGES) n = MAX_MESSAGES;
    req->msg_roles = calloc(n, sizeof(char *));
    req->msg_contents = calloc(n, sizeof(char *));
    req->msg_tool_call_ids = calloc(n, sizeof(char *));
    req->num_messages = n;

    for (int i = 0; i < n; i++) {
        NSDictionary *msg = messages[i];
        if (![msg isKindOfClass:[NSDictionary class]]) continue;

        NSString *role = msg[@"role"];
        req->msg_roles[i] = role ? strdup([role UTF8String]) : strdup("user");

        // Content can be string or array
        id content = msg[@"content"];
        if ([content isKindOfClass:[NSString class]]) {
            req->msg_contents[i] = strdup([content UTF8String]);
        } else if ([content isKindOfClass:[NSArray class]]) {
            // Multimodal content — extract text parts
            NSMutableString *text = [NSMutableString string];
            for (NSDictionary *part in content) {
                if ([part isKindOfClass:[NSDictionary class]] &&
                    [part[@"type"] isEqualToString:@"text"]) {
                    [text appendString:part[@"text"] ?: @""];
                }
            }
            req->msg_contents[i] = strdup([text UTF8String]);
        } else {
            req->msg_contents[i] = strdup("");
        }

        NSString *tcid = msg[@"tool_call_id"];
        req->msg_tool_call_ids[i] = tcid ? strdup([tcid UTF8String]) : NULL;
    }

    // Sampling parameters
    NSNumber *temp = json[@"temperature"];
    if (temp) req->temperature = [temp floatValue];
    if (req->temperature < 0) req->temperature = 0;
    if (req->temperature > 2.0f) req->temperature = 2.0f;

    NSNumber *top_p = json[@"top_p"];
    if (top_p) req->top_p = [top_p floatValue];
    if (req->top_p < 0) req->top_p = 0;
    if (req->top_p > 1.0f) req->top_p = 1.0f;

    NSNumber *top_k_val = json[@"top_k"];
    if (top_k_val) req->top_k = [top_k_val intValue];

    NSNumber *max_tok = json[@"max_tokens"];
    if (!max_tok) max_tok = json[@"max_completion_tokens"];
    if (max_tok) req->max_tokens = [max_tok intValue];
    if (req->max_tokens <= 0) req->max_tokens = 4096;
    if (req->max_tokens > MAX_GEN_TOKENS) req->max_tokens = MAX_GEN_TOKENS;

    NSNumber *min_tok = json[@"min_tokens"];
    if (min_tok) req->min_tokens = [min_tok intValue];
    if (req->min_tokens < 0) req->min_tokens = 0;
    if (req->min_tokens > req->max_tokens) req->min_tokens = req->max_tokens;

    NSNumber *freq_pen = json[@"frequency_penalty"];
    if (freq_pen) req->frequency_penalty = [freq_pen floatValue];

    NSNumber *pres_pen = json[@"presence_penalty"];
    if (pres_pen) req->presence_penalty = [pres_pen floatValue];

    // Stream
    NSNumber *stream_val = json[@"stream"];
    if (stream_val) req->stream = [stream_val boolValue];

    NSDictionary *stream_opts = json[@"stream_options"];
    if (stream_opts && [stream_opts isKindOfClass:[NSDictionary class]]) {
        NSNumber *inc_usage = stream_opts[@"include_usage"];
        if (inc_usage) req->include_usage = [inc_usage boolValue];
    }

    // Session-based KV caching
    NSNumber *cache_val = json[@"cache"];
    if (cache_val) req->cache = [cache_val boolValue];
    NSString *sid = json[@"session_id"];
    if (sid && [sid isKindOfClass:[NSString class]]) {
        strlcpy(req->session_id, [sid UTF8String], sizeof(req->session_id));
    }

    // Stop sequences
    id stop = json[@"stop"];
    if ([stop isKindOfClass:[NSString class]]) {
        strlcpy(req->stop_seqs[0], [stop UTF8String], MAX_STOP_SEQ_LEN);
        req->num_stop_seqs = 1;
    } else if ([stop isKindOfClass:[NSArray class]]) {
        for (NSString *s in stop) {
            if (req->num_stop_seqs >= MAX_STOP_SEQUENCES) break;
            if ([s isKindOfClass:[NSString class]]) {
                strlcpy(req->stop_seqs[req->num_stop_seqs], [s UTF8String], MAX_STOP_SEQ_LEN);
                req->num_stop_seqs++;
            }
        }
    }

    // Tools
    NSArray *tools = json[@"tools"];
    if (tools && [tools isKindOfClass:[NSArray class]] && [tools count] > 0) {
        NSData *td = [NSJSONSerialization dataWithJSONObject:tools options:0 error:nil];
        if (td) {
            req->tools_json = calloc([td length] + 1, 1);
            memcpy(req->tools_json, [td bytes], [td length]);
        }
    }

    NSString *tc = nil;
    id tool_choice = json[@"tool_choice"];
    if ([tool_choice isKindOfClass:[NSString class]]) {
        tc = tool_choice;
    } else if ([tool_choice isKindOfClass:[NSDictionary class]]) {
        tc = @"auto";
    }
    req->tool_choice = tc ? strdup([tc UTF8String]) : NULL;

    // Generate request ID and timestamp
    generate_request_id(req->request_id, sizeof(req->request_id));
    struct timeval tv;
    gettimeofday(&tv, NULL);
    req->created = tv.tv_sec;

    return 0;
}

// ============================================================================
// Chat template builder
// ============================================================================

// Build the tools system block for Qwen3.5 chat template.
// Returns malloc'd string with the tools definition formatted for the model,
// or NULL if no tools are provided.
// Caller must free.
static char *build_tools_block(CompletionRequest *req) {
    if (!req->tools_json || strlen(req->tools_json) == 0) return NULL;

    // Qwen3.5 expects tools in a system message with this format:
    //   <|im_start|>system
    //   # Tools
    //   You are provided with function signatures ...
    //   <tools>
    //   [{"type":"function","function":{...}}, ...]
    //   </tools>
    //   ... instructions on how to call them ...
    //   <|im_end|>

    const char *tools_preamble =
        "\n\n# Tools\n\n"
        "You are provided with function signatures within <tools></tools> XML tags.\n\n"
        "<tools>\n";
    const char *tools_postamble =
        "\n</tools>\n\n"
        "For each function call, return a JSON object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        "{\"name\": \"<function-name>\", \"arguments\": <args-json-object>}\n"
        "</tool_call>";

    size_t tools_len = strlen(req->tools_json);
    size_t total = strlen(tools_preamble) + tools_len + strlen(tools_postamble) + 1;
    char *block = malloc(total);
    int pos = 0;
    pos += snprintf(block + pos, total - pos, "%s", tools_preamble);
    pos += snprintf(block + pos, total - pos, "%s", req->tools_json);
    pos += snprintf(block + pos, total - pos, "%s", tools_postamble);
    return block;
}

// Build the user turn portion of the Qwen3.5 chat template.
// The system prompt is already cached in the snapshot — we only build
// the user/assistant/tool messages that come AFTER it.
// When tools are provided, injects the tools definition as a continuation
// of the cached system prompt (appended before the first user message).
// Returns malloc'd string, caller must free.
static char *build_user_turn(CompletionRequest *req) {
    int first_user = 0;

    // Build tools block if tools are provided
    char *tools_block = build_tools_block(req);

    // Calculate buffer size
    size_t total = 0;
    for (int i = first_user; i < req->num_messages; i++) {
        total += 50 + strlen(req->msg_contents[i]);
    }
    total += 100; // padding
    if (tools_block) total += strlen(tools_block) + 200;

    char *buf = malloc(total);
    int pos = 0;

    // Inject tools definition as system block if provided
    if (tools_block) {
        pos += snprintf(buf + pos, total - pos,
            "<|im_start|>system\n%s<|im_end|>\n", tools_block);
        free(tools_block);
    }

    // Build each message in chat template format
    int last = req->num_messages - 1;
    for (int i = first_user; i < req->num_messages; i++) {
        const char *role = req->msg_roles[i];
        const char *content = req->msg_contents[i];

        if (strcmp(role, "tool") == 0) {
            pos += snprintf(buf + pos, total - pos,
                "<|im_start|>user\n<tool_response>\n%s\n</tool_response><|im_end|>\n",
                content);
        } else if (i == last && strcmp(role, "assistant") == 0) {
            pos += snprintf(buf + pos, total - pos,
                "<|im_start|>assistant\n%s", content);
        } else {
            pos += snprintf(buf + pos, total - pos,
                "<|im_start|>%s\n%s<|im_end|>\n", role, content);
        }
    }

    // If last message was NOT assistant, add the assistant prompt
    if (last < first_user || strcmp(req->msg_roles[last], "assistant") != 0) {
        pos += snprintf(buf + pos, total - pos, "<|im_start|>assistant\n");
    }

    return buf;
}

// ============================================================================
// Sampling
// ============================================================================

// Comparison function for sorting (index, prob) pairs by prob descending
typedef struct { int idx; float prob; } IdxProb;
static int cmp_idx_prob_desc(const void *a, const void *b) {
    float pa = ((const IdxProb *)a)->prob;
    float pb = ((const IdxProb *)b)->prob;
    if (pb > pa) return 1;
    if (pb < pa) return -1;
    return 0;
}

static int sample_token(float *logits, int vocab_size, CompletionRequest *req,
                        int *token_counts, int num_generated) {
    (void)num_generated;

    // Apply frequency and presence penalties
    if (req->frequency_penalty != 0.0f || req->presence_penalty != 0.0f) {
        for (int i = 0; i < vocab_size; i++) {
            if (token_counts[i] > 0) {
                logits[i] -= req->presence_penalty;
                logits[i] -= req->frequency_penalty * token_counts[i];
            }
        }
    }

    // Temperature = 0 → greedy
    if (req->temperature < 1e-6f) {
        return cpu_argmax(logits, vocab_size);
    }

    // Scale by temperature
    float inv_temp = 1.0f / req->temperature;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] *= inv_temp;
    }

    // Top-K filtering
    int effective_k = vocab_size;
    if (req->top_k > 0 && req->top_k < vocab_size) {
        effective_k = req->top_k;
        // Find k-th largest using partial sort
        // For efficiency, use a simple approach: find the k-th value threshold
        float *tmp = malloc(vocab_size * sizeof(float));
        memcpy(tmp, logits, vocab_size * sizeof(float));

        // Partial sort to find threshold (nth_element equivalent)
        // Use a simpler approach: sort a copy and find threshold
        // For vocab_size ~248K and top_k typically 40-100, this is fast enough
        for (int i = 0; i < effective_k; i++) {
            int max_idx = i;
            for (int j = i + 1; j < vocab_size; j++) {
                if (tmp[j] > tmp[max_idx]) max_idx = j;
            }
            if (max_idx != i) {
                float t = tmp[i]; tmp[i] = tmp[max_idx]; tmp[max_idx] = t;
            }
        }
        float threshold = tmp[effective_k - 1];
        free(tmp);

        for (int i = 0; i < vocab_size; i++) {
            if (logits[i] < threshold) logits[i] = -INFINITY;
        }
    }

    // Softmax
    float max_val = -INFINITY;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > -1e30f) {
            logits[i] = expf(logits[i] - max_val);
            sum += logits[i];
        } else {
            logits[i] = 0.0f;
        }
    }
    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < vocab_size; i++) logits[i] *= inv_sum;
    }

    // Top-P (nucleus) sampling
    if (req->top_p < 1.0f && req->top_p > 0.0f) {
        // Collect non-zero probabilities
        IdxProb *pairs = malloc(vocab_size * sizeof(IdxProb));
        int n_active = 0;
        for (int i = 0; i < vocab_size; i++) {
            if (logits[i] > 0.0f) {
                pairs[n_active].idx = i;
                pairs[n_active].prob = logits[i];
                n_active++;
            }
        }

        // Sort descending
        qsort(pairs, n_active, sizeof(IdxProb), cmp_idx_prob_desc);

        // Find cutoff
        float cumsum = 0.0f;
        int cutoff = n_active;
        for (int i = 0; i < n_active; i++) {
            cumsum += pairs[i].prob;
            if (cumsum >= req->top_p) {
                cutoff = i + 1;
                break;
            }
        }

        // Zero out everything past cutoff
        for (int i = cutoff; i < n_active; i++) {
            logits[pairs[i].idx] = 0.0f;
        }
        free(pairs);

        // Renormalize
        sum = 0.0f;
        for (int i = 0; i < vocab_size; i++) sum += logits[i];
        if (sum > 0.0f) {
            float inv = 1.0f / sum;
            for (int i = 0; i < vocab_size; i++) logits[i] *= inv;
        }
    }

    // Weighted random selection
    float r = (float)arc4random() / (float)UINT32_MAX;
    float cumulative = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumulative += logits[i];
        if (cumulative >= r) return i;
    }

    // Fallback: return last non-zero
    return cpu_argmax(logits, vocab_size);
}

// ============================================================================
// Stop sequence detector
// ============================================================================

typedef struct {
    char buffer[1024];  // buffered output not yet emitted
    int  buf_len;
} StopDetector;

// Feed a token to the stop detector.
// Returns: 0=no match (safe to emit), 1=stop found, 2=partial match (buffer but don't emit yet)
static int stop_check(StopDetector *sd, const char *token, CompletionRequest *req,
                      char *emit_buf, int emit_buf_size) {
    if (req->num_stop_seqs == 0) {
        // No stop sequences — just pass through
        strlcpy(emit_buf, token, emit_buf_size);
        return 0;
    }

    // Append token to buffer
    int tlen = (int)strlen(token);
    if (sd->buf_len + tlen < (int)sizeof(sd->buffer) - 1) {
        memcpy(sd->buffer + sd->buf_len, token, tlen);
        sd->buf_len += tlen;
        sd->buffer[sd->buf_len] = '\0';
    }

    // Check for full matches
    for (int s = 0; s < req->num_stop_seqs; s++) {
        char *found = strstr(sd->buffer, req->stop_seqs[s]);
        if (found) {
            // Stop found — emit everything before the match
            int pre_len = (int)(found - sd->buffer);
            if (pre_len > 0 && pre_len < emit_buf_size) {
                memcpy(emit_buf, sd->buffer, pre_len);
                emit_buf[pre_len] = '\0';
            } else {
                emit_buf[0] = '\0';
            }
            return 1;
        }
    }

    // Check for partial matches at the end of buffer
    for (int s = 0; s < req->num_stop_seqs; s++) {
        int slen = (int)strlen(req->stop_seqs[s]);
        for (int overlap = 1; overlap < slen && overlap <= sd->buf_len; overlap++) {
            if (memcmp(sd->buffer + sd->buf_len - overlap, req->stop_seqs[s], overlap) == 0) {
                // Partial match — emit everything before the potential match
                int safe_len = sd->buf_len - overlap;
                if (safe_len > 0 && safe_len < emit_buf_size) {
                    memcpy(emit_buf, sd->buffer, safe_len);
                    emit_buf[safe_len] = '\0';
                    // Shift buffer
                    memmove(sd->buffer, sd->buffer + safe_len, overlap);
                    sd->buf_len = overlap;
                    sd->buffer[sd->buf_len] = '\0';
                } else {
                    emit_buf[0] = '\0';
                }
                return 2; // partial match, buffered
            }
        }
    }

    // No match — emit entire buffer
    if (sd->buf_len > 0 && sd->buf_len < emit_buf_size) {
        memcpy(emit_buf, sd->buffer, sd->buf_len);
        emit_buf[sd->buf_len] = '\0';
        sd->buf_len = 0;
        sd->buffer[0] = '\0';
    } else {
        emit_buf[0] = '\0';
    }
    return 0;
}

// ============================================================================
// Tool call parser (XML <tool_call> → OpenAI JSON)
// ============================================================================

typedef struct {
    char id[32];
    char name[128];
    char arguments[8192];
} ParsedToolCall;

static int parse_tool_calls(const char *text, ParsedToolCall *calls, int max_calls) {
    int count = 0;
    const char *p = text;

    while (count < max_calls) {
        const char *start = strstr(p, "<tool_call>");
        if (!start) break;
        start += 11; // skip "<tool_call>"

        const char *end = strstr(start, "</tool_call>");
        if (!end) break;

        // Extract JSON between tags
        int json_len = (int)(end - start);
        char *json_str = malloc(json_len + 1);
        memcpy(json_str, start, json_len);
        json_str[json_len] = '\0';

        // Parse with NSJSONSerialization
        NSData *jdata = [NSData dataWithBytes:json_str length:json_len];
        NSDictionary *obj = [NSJSONSerialization JSONObjectWithData:jdata options:0 error:nil];
        free(json_str);

        if (obj && [obj isKindOfClass:[NSDictionary class]]) {
            generate_call_id(calls[count].id, sizeof(calls[count].id));

            NSString *name = obj[@"name"];
            if (name) strlcpy(calls[count].name, [name UTF8String], sizeof(calls[count].name));

            id args = obj[@"arguments"];
            if (args) {
                if ([args isKindOfClass:[NSDictionary class]] || [args isKindOfClass:[NSArray class]]) {
                    NSData *ad = [NSJSONSerialization dataWithJSONObject:args options:0 error:nil];
                    if (ad && [ad length] < sizeof(calls[count].arguments)) {
                        memcpy(calls[count].arguments, [ad bytes], [ad length]);
                        calls[count].arguments[[ad length]] = '\0';
                    }
                } else if ([args isKindOfClass:[NSString class]]) {
                    strlcpy(calls[count].arguments, [args UTF8String], sizeof(calls[count].arguments));
                }
            }
            count++;
        }

        p = end + 12; // skip "</tool_call>"
    }

    return count;
}

// ============================================================================
// SSE streaming helpers
// ============================================================================

static void sse_send_headers(int fd) {
    const char *headers =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n";
    http_write_str(fd, headers);
}

// Send the first chunk with role:"assistant"
static int sse_send_role(int fd, CompletionRequest *req) {
    char buf[512];
    int n = snprintf(buf, sizeof(buf),
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"created\":%ld,\"model\":\"%s\","
        "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},"
        "\"finish_reason\":null}]}\n\n",
        req->request_id, req->created,
        req->model[0] ? req->model : "flash-moe");
    ssize_t w = write(fd, buf, n);
    return w > 0 ? 0 : -1;
}

// ============================================================================
// UTF-8 streaming buffer
// ============================================================================
// BPE tokens can split multi-byte UTF-8 sequences across token boundaries
// (e.g., emoji 👋 = F0 9F 91 8B may be two tokens: [F0 9F] [91 8B]).
// Each fragment alone is invalid UTF-8, so we buffer partial sequences and
// only emit complete UTF-8 codepoints.

typedef struct {
    char  pending[8];   // at most 3 trailing bytes of incomplete sequence
    int   pending_len;
} Utf8StreamBuf;

// Returns number of bytes from the END of buf that form an incomplete UTF-8 sequence.
// 0 means the entire buffer is valid UTF-8.
static int utf8_incomplete_tail(const char *buf, int len) {
    if (len == 0) return 0;
    // Scan backwards from the end to find the start of the last codepoint
    for (int i = 1; i <= 4 && i <= len; i++) {
        unsigned char c = (unsigned char)buf[len - i];
        if ((c & 0x80) == 0) {
            // ASCII — the last char is complete
            return 0;
        }
        if ((c & 0xC0) == 0xC0) {
            // Start byte found — check if the sequence is complete
            int expected;
            if ((c & 0xE0) == 0xC0) expected = 2;
            else if ((c & 0xF0) == 0xE0) expected = 3;
            else if ((c & 0xF8) == 0xF0) expected = 4;
            else return i; // invalid start byte, treat as incomplete
            if (i >= expected) return 0;  // sequence is complete
            return i;  // sequence is incomplete
        }
        // continuation byte (10xxxxxx) — keep scanning back
    }
    // All continuation bytes, no start byte found within 4 bytes — all incomplete
    return len < 4 ? len : 4;
}

// Append data to the UTF-8 buffer, return pointer to a complete string to emit
// (may be empty ""). The returned pointer is valid until the next call.
// The caller should emit the returned string if it's non-empty.
static const char *utf8_stream_push(Utf8StreamBuf *u, const char *data, int len,
                                     char *out, int out_size) {
    // Prepend any pending bytes from last call
    int total = u->pending_len + len;
    if (total >= out_size - 1) total = out_size - 2;

    if (u->pending_len > 0) {
        memcpy(out, u->pending, u->pending_len);
    }
    int copy_len = total - u->pending_len;
    if (copy_len > 0) memcpy(out + u->pending_len, data, copy_len);
    out[total] = '\0';

    // Check if the tail is an incomplete UTF-8 sequence
    int tail = utf8_incomplete_tail(out, total);
    if (tail > 0) {
        // Save incomplete tail for next call
        memcpy(u->pending, out + total - tail, tail);
        u->pending_len = tail;
        out[total - tail] = '\0';
    } else {
        u->pending_len = 0;
    }
    return out;
}

// Flush any remaining pending bytes (called at end of generation)
static const char *utf8_stream_flush(Utf8StreamBuf *u, char *out, int out_size) {
    if (u->pending_len > 0 && u->pending_len < out_size - 1) {
        memcpy(out, u->pending, u->pending_len);
        out[u->pending_len] = '\0';
        u->pending_len = 0;
        return out;
    }
    out[0] = '\0';
    return out;
}

// Send a content delta
static int sse_send_delta(int fd, CompletionRequest *req, const char *content) {
    if (!content || !content[0]) return 0;

    char escaped[4096];
    json_escape(content, escaped, sizeof(escaped));

    char buf[8192];
    int n = snprintf(buf, sizeof(buf),
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"created\":%ld,\"model\":\"%s\","
        "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},"
        "\"finish_reason\":null}]}\n\n",
        req->request_id, req->created,
        req->model[0] ? req->model : "flash-moe",
        escaped);
    if (n >= (int)sizeof(buf)) n = (int)sizeof(buf) - 1;
    ssize_t w = write(fd, buf, n);
    return w > 0 ? 0 : -1;
}

// Send tool call deltas for streaming (one chunk per tool call)
static void sse_send_tool_calls(int fd, CompletionRequest *req,
                                 ParsedToolCall *tool_calls, int num_tool_calls) {
    for (int i = 0; i < num_tool_calls; i++) {
        char escaped_args[8192];
        json_escape(tool_calls[i].arguments, escaped_args, sizeof(escaped_args));

        char buf[16384];
        int n = snprintf(buf, sizeof(buf),
            "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
            "\"created\":%ld,\"model\":\"%s\","
            "\"choices\":[{\"index\":0,\"delta\":{"
            "\"tool_calls\":[{\"index\":%d,\"id\":\"%s\",\"type\":\"function\","
            "\"function\":{\"name\":\"%s\",\"arguments\":\"%s\"}}]},"
            "\"finish_reason\":null}]}\n\n",
            req->request_id, req->created,
            req->model[0] ? req->model : "flash-moe",
            i, tool_calls[i].id, tool_calls[i].name, escaped_args);
        write(fd, buf, n);
    }
}

// Send finish chunk
static void sse_send_finish(int fd, CompletionRequest *req, const char *finish_reason,
                             int prompt_tokens, int completion_tokens, int cached_tokens,
                             double prefill_tps, double decode_tps) {
    char buf[1024];

    if (req->include_usage) {
        snprintf(buf, sizeof(buf),
            "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
            "\"created\":%ld,\"model\":\"%s\","
            "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}],"
            "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,"
            "\"cached_tokens\":%d,\"prefill_tokens_per_second\":%.2f,\"decode_tokens_per_second\":%.2f}}\n\n",
            req->request_id, req->created,
            req->model[0] ? req->model : "flash-moe",
            finish_reason,
            prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
            cached_tokens, prefill_tps, decode_tps);
    } else {
        snprintf(buf, sizeof(buf),
            "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
            "\"created\":%ld,\"model\":\"%s\","
            "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}]}\n\n",
            req->request_id, req->created,
            req->model[0] ? req->model : "flash-moe",
            finish_reason);
    }
    http_write_str(fd, buf);
}

static void sse_send_done(int fd) {
    http_write_str(fd, "data: [DONE]\n\n");
}

// ============================================================================
// Non-streaming response builder
// ============================================================================

static char *build_completion_response(CompletionRequest *req, const char *content,
                                        ParsedToolCall *tool_calls, int num_tool_calls,
                                        const char *finish_reason,
                                        int prompt_tokens, int completion_tokens,
                                        int cached_tokens,
                                        double prefill_tps, double decode_tps) {
    // Build message object
    char *buf = malloc(RESPONSE_BUF_SIZE);
    int pos = 0;

    pos += snprintf(buf + pos, RESPONSE_BUF_SIZE - pos,
        "{\"id\":\"%s\",\"object\":\"chat.completion\","
        "\"created\":%ld,\"model\":\"%s\","
        "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\"",
        req->request_id, req->created,
        req->model[0] ? req->model : "flash-moe");

    if (num_tool_calls > 0) {
        // Content before tool calls (or null)
        if (content && strlen(content) > 0) {
            char escaped[RESPONSE_BUF_SIZE / 2];
            json_escape(content, escaped, sizeof(escaped));
            pos += snprintf(buf + pos, RESPONSE_BUF_SIZE - pos, ",\"content\":\"%s\"", escaped);
        } else {
            pos += snprintf(buf + pos, RESPONSE_BUF_SIZE - pos, ",\"content\":null");
        }

        // Tool calls array
        pos += snprintf(buf + pos, RESPONSE_BUF_SIZE - pos, ",\"tool_calls\":[");
        for (int i = 0; i < num_tool_calls; i++) {
            if (i > 0) pos += snprintf(buf + pos, RESPONSE_BUF_SIZE - pos, ",");

            char escaped_args[8192];
            json_escape(tool_calls[i].arguments, escaped_args, sizeof(escaped_args));

            pos += snprintf(buf + pos, RESPONSE_BUF_SIZE - pos,
                "{\"id\":\"%s\",\"type\":\"function\","
                "\"function\":{\"name\":\"%s\",\"arguments\":\"%s\"}}",
                tool_calls[i].id, tool_calls[i].name, escaped_args);
        }
        pos += snprintf(buf + pos, RESPONSE_BUF_SIZE - pos, "]");
    } else {
        char escaped[RESPONSE_BUF_SIZE / 2];
        json_escape(content ? content : "", escaped, sizeof(escaped));
        pos += snprintf(buf + pos, RESPONSE_BUF_SIZE - pos, ",\"content\":\"%s\"", escaped);
    }

    pos += snprintf(buf + pos, RESPONSE_BUF_SIZE - pos,
        "},\"finish_reason\":\"%s\"}],"
        "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,"
        "\"cached_tokens\":%d,\"prefill_tokens_per_second\":%.2f,\"decode_tokens_per_second\":%.2f}}",
        finish_reason,
        prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
        cached_tokens, prefill_tps, decode_tps);

    return buf;
}

// ============================================================================
// Request queue
// ============================================================================

typedef struct {
    int   client_fd;
    char *body;
    int   body_len;
} QueuedRequest;

typedef struct {
    QueuedRequest   items[REQUEST_QUEUE_SIZE];
    int             head, tail, count;
    pthread_mutex_t mutex;
    pthread_cond_t  not_empty;
    pthread_cond_t  not_full;
} RequestQueue;

static void queue_init(RequestQueue *q) {
    memset(q, 0, sizeof(*q));
    pthread_mutex_init(&q->mutex, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
}

static int queue_push(RequestQueue *q, int client_fd, char *body, int body_len) {
    pthread_mutex_lock(&q->mutex);
    while (q->count >= REQUEST_QUEUE_SIZE) {
        pthread_cond_wait(&q->not_full, &q->mutex);
    }
    q->items[q->tail].client_fd = client_fd;
    q->items[q->tail].body = body;
    q->items[q->tail].body_len = body_len;
    q->tail = (q->tail + 1) % REQUEST_QUEUE_SIZE;
    q->count++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mutex);
    return 0;
}

static QueuedRequest queue_pop(RequestQueue *q) {
    pthread_mutex_lock(&q->mutex);
    while (q->count == 0) {
        pthread_cond_wait(&q->not_empty, &q->mutex);
    }
    QueuedRequest r = q->items[q->head];
    q->head = (q->head + 1) % REQUEST_QUEUE_SIZE;
    q->count--;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mutex);
    return r;
}

// ============================================================================
// Inference worker thread
// ============================================================================

typedef struct {
    InferContext   *ctx;
    RequestQueue   *queue;
    InferStateSnapshot *sys_snapshot;
    int             sys_pos;
} InferWorkerArgs;

static void *inference_worker(void *arg) {
    InferWorkerArgs *wa = (InferWorkerArgs *)arg;
    InferContext *ctx = wa->ctx;
    RequestQueue *queue = wa->queue;
    InferStateSnapshot *sys_snap = wa->sys_snapshot;
    int sys_pos = wa->sys_pos;

    // Allocate token count buffer for penalties
    int *token_counts = calloc(cfg.vocab_size, sizeof(int));

    // Session-based KV cache — continues from where last generation ended
    // (same approach as serve_loop in infer.m: no snapshot/restore, just
    //  keep the KV state and track the position)
    char active_session[64] = {0};
    int  session_pos = 0;

    fprintf(stderr, "[worker] Inference worker ready (sys_pos=%d)\n", sys_pos);

    for (;;) {
        @autoreleasepool {
        QueuedRequest qr = queue_pop(queue);
        int client_fd = qr.client_fd;

        // Parse request
        CompletionRequest req;
        if (parse_completion_request(qr.body, qr.body_len, &req) < 0) {
            send_error(client_fd, 400, "Invalid request JSON", "invalid_request_error");
            free(qr.body);
            close(client_fd);
            continue;
        }
        free(qr.body);

        fprintf(stderr, "[worker] %s messages=%d max_tokens=%d min_tokens=%d temp=%.2f stream=%d\n",
                req.request_id, req.num_messages, req.max_tokens, req.min_tokens,
                req.temperature, req.stream);

        // Session-based KV caching check
        int is_continuation = (req.cache && req.session_id[0] &&
                               active_session[0] &&
                               strcmp(req.session_id, active_session) == 0);

        // Build template text.
        // Continuation: client sends only the new user message, we wrap it
        // with \n<|im_start|>user\n ... <|im_end|>\n<|im_start|>assistant\n
        // (EOS/im_end from previous generation is already in KV cache).
        // New session: full template from all messages.
        char *template_text;
        if (is_continuation) {
            // Extract last user content from messages array
            const char *user_content = "";
            for (int i = req.num_messages - 1; i >= 0; i--) {
                if (strcmp(req.msg_roles[i], "user") == 0) {
                    user_content = req.msg_contents[i];
                    break;
                }
            }
            // Build continuation template (same as tokenize_continuation_turn in infer.m)
            size_t total = 200 + strlen(user_content);

            // Check if last message is an assistant prefill (thinking disabled)
            const char *assistant_suffix = "";
            if (req.num_messages > 0 &&
                strcmp(req.msg_roles[req.num_messages - 1], "assistant") == 0) {
                assistant_suffix = req.msg_contents[req.num_messages - 1];
            }
            total += strlen(assistant_suffix);

            template_text = malloc(total);
            snprintf(template_text, total,
                "\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n%s",
                user_content, assistant_suffix);
        } else {
            template_text = build_user_turn(&req);
        }

        if (!template_text) {
            send_error(client_fd, 500, "Failed to build chat template", "server_error");
            free_completion_request(&req);
            close(client_fd);
            continue;
        }

        // Tokenize
        PromptTokens *pt = infer_encode_text(template_text);
        free(template_text);
        if (!pt || pt->count == 0) {
            send_error(client_fd, 500, "Tokenization failed", "server_error");
            free_completion_request(&req);
            close(client_fd);
            continue;
        }

        int prompt_tokens = pt->count;
        int cached_tokens = 0;

        int pos;
        if (is_continuation) {
            pos = session_pos;
            cached_tokens = pos - sys_pos;
            fprintf(stderr, "[worker] %s prompt_tokens=%d session=%s [CONTINUE pos=%d]\n",
                    req.request_id, prompt_tokens, req.session_id, pos);
        } else {
            // New session or no cache — restore system prompt snapshot
            infer_restore_state(ctx, sys_snap);
            pos = sys_pos;
            if (req.cache && req.session_id[0]) {
                strlcpy(active_session, req.session_id, sizeof(active_session));
            } else {
                active_session[0] = '\0';
            }
            fprintf(stderr, "[worker] %s prompt_tokens=%d session=%s [NEW]\n",
                    req.request_id, prompt_tokens,
                    req.session_id[0] ? req.session_id : "(none)");
        }

        // Prefill prompt tokens (with abort check)
        // Fast prefill: skip routed expert I/O for intermediate tokens (shared expert only)
        double t_prefill = now_ms();
        int aborted = 0;
        for (int i = 0; i < pt->count - 1; i++) { @autoreleasepool {
            if ((i & 3) == 0 && client_disconnected(client_fd)) {
                fprintf(stderr, "[worker] %s client disconnected during prefill at token %d/%d\n",
                        req.request_id, i, pt->count);
                aborted = 1;
                break;
            }
            infer_embed_token(ctx, pt->ids[i]);
            for (int layer = 0; layer < cfg.num_layers; layer++) {
                infer_forward_layer(ctx, layer, pos);
            }
            infer_discard_deferred_nowait();
            pos++;
        } /* @autoreleasepool */ }
        if (aborted) {
            infer_free_tokens(pt);
            free_completion_request(&req);
            close(client_fd);
            continue;
        }
        // Last prompt token — full completion
        infer_embed_token(ctx, pt->ids[pt->count - 1]);
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            infer_forward_layer(ctx, layer, pos);
        }
        infer_complete_deferred();
        pos++;

        infer_final_norm(ctx);
        infer_lm_head(ctx);

        double prefill_ms = now_ms() - t_prefill;
        double prefill_tps = prompt_tokens > 0 && prefill_ms > 0 ? prompt_tokens * 1000.0 / prefill_ms : 0.0;
        fprintf(stderr, "[worker] %s prefill=%d tokens (cached=%d) in %.0fms (%.1f t/s)\n",
                req.request_id, pt->count, cached_tokens, prefill_ms, prefill_tps);
        infer_free_tokens(pt);

        // Start streaming or accumulating
        if (req.stream) {
            sse_send_headers(client_fd);
            sse_send_role(client_fd, &req);
        }

        // Reset penalty tracking
        memset(token_counts, 0, cfg.vocab_size * sizeof(int));

        // Generation loop
        Utf8StreamBuf u8buf = {0};
        double t_gen = now_ms();
        int gen_count = 0;
        int in_think = 0;
        int think_tokens = 0;
        int think_budget = 2048;
        const char *finish_reason = "stop";

        // Accumulate full response (for non-streaming + tool call detection)
        char *response_buf = calloc(1, RESPONSE_BUF_SIZE);
        int resp_len = 0;

        StopDetector sd = {0};

        // Make a copy of logits for sampling (sample_token modifies in-place)
        float *logits_copy = malloc(cfg.vocab_size * sizeof(float));

        for (int gen = 0; gen < req.max_tokens; gen++) { @autoreleasepool {
            // Abort if client disconnected (check every 8 tokens to avoid syscall overhead)
            if ((gen & 7) == 0 && client_disconnected(client_fd)) {
                fprintf(stderr, "[worker] %s client disconnected during generation at token %d\n",
                        req.request_id, gen);
                finish_reason = "abort";
                break;
            }

            // Sample next token
            memcpy(logits_copy, infer_get_logits(ctx), cfg.vocab_size * sizeof(float));

            // min_tokens: suppress EOS logits before sampling so model can't stop early
            if (req.min_tokens > 0 && gen < req.min_tokens) {
                for (int e = 0; e < cfg.num_eos_tokens; e++) {
                    logits_copy[cfg.eos_token_ids[e]] = -INFINITY;
                }
            }

            int next_token = sample_token(logits_copy, cfg.vocab_size, &req, token_counts, gen);

            // EOS check
            if (infer_is_eos(next_token)) {
                // Feed EOS through the model so session KV state includes it
                // (required for session-based caching to work on continuation)
                if (req.cache && req.session_id[0]) {
                    infer_embed_token(ctx, next_token);
                    for (int layer = 0; layer < cfg.num_layers; layer++) {
                        infer_forward_layer(ctx, layer, pos);
                    }
                    infer_discard_deferred();
                    pos++;
                }
                finish_reason = "stop";
                break;
            }

            // Think budget
            if (next_token == cfg.think_start_token) in_think = 1;
            if (next_token == cfg.think_end_token) { in_think = 0; think_tokens = 0; }
            if (in_think) {
                think_tokens++;
                if (think_budget > 0 && think_tokens >= think_budget) {
                    next_token = cfg.think_end_token;
                    in_think = 0;
                }
            }

            // Decode token
            const char *tok_str = infer_decode_token(ctx, next_token);
            token_counts[next_token]++;
            gen_count++;

            // Accumulate response
            int tlen = (int)strlen(tok_str);
            if (resp_len + tlen < RESPONSE_BUF_SIZE - 1) {
                memcpy(response_buf + resp_len, tok_str, tlen);
                resp_len += tlen;
                response_buf[resp_len] = '\0';
            }

            // Streaming output
            int under_min = (req.min_tokens > 0 && gen < req.min_tokens);
            if (req.stream) {
                // Check stop sequences (skip stopping if under min_tokens)
                char emit[4096];
                int stop_result = under_min ? 0 : stop_check(&sd, tok_str, &req, emit, sizeof(emit));
                if (under_min) {
                    // Under min_tokens — bypass stop detector, emit directly
                    strncpy(emit, tok_str, sizeof(emit) - 1);
                    emit[sizeof(emit) - 1] = '\0';
                }
                if (stop_result == 1) {
                    // Stop found — emit any pre-match content through UTF-8 buffer
                    if (emit[0]) {
                        char u8out[4096];
                        const char *safe = utf8_stream_push(&u8buf, emit, (int)strlen(emit), u8out, sizeof(u8out));
                        if (safe[0]) sse_send_delta(client_fd, &req, safe);
                        // Flush any remaining bytes
                        safe = utf8_stream_flush(&u8buf, u8out, sizeof(u8out));
                        if (safe[0]) sse_send_delta(client_fd, &req, safe);
                    }
                    finish_reason = "stop";
                    break;
                } else if ((stop_result == 0 || under_min) && emit[0]) {
                    // No match — emit through UTF-8 buffer
                    char u8out[4096];
                    const char *safe = utf8_stream_push(&u8buf, emit, (int)strlen(emit), u8out, sizeof(u8out));
                    if (safe[0]) {
                        if (sse_send_delta(client_fd, &req, safe) < 0) {
                            fprintf(stderr, "[worker] %s client disconnected\n", req.request_id);
                            finish_reason = "abort";
                            break;
                        }
                    }
                }
                // stop_result == 2: partial match, buffered — don't emit
            } else {
                // Non-streaming: check disconnect periodically
                if ((gen & 15) == 0 && client_disconnected(client_fd)) {
                    fprintf(stderr, "[worker] %s client disconnected (non-streaming)\n", req.request_id);
                    finish_reason = "abort";
                    goto gen_done;
                }
                // Non-streaming stop check (skip if under min_tokens)
                if (!under_min) {
                    for (int s = 0; s < req.num_stop_seqs; s++) {
                        if (strstr(response_buf, req.stop_seqs[s])) {
                            char *found = strstr(response_buf, req.stop_seqs[s]);
                            *found = '\0';
                            resp_len = (int)(found - response_buf);
                            finish_reason = "stop";
                            goto gen_done;
                        }
                    }
                }
            }

            // Check max tokens
            if (gen == req.max_tokens - 1) {
                finish_reason = "length";
                break;
            }

            // Forward next token
            infer_embed_token(ctx, next_token);
            for (int layer = 0; layer < cfg.num_layers; layer++) {
                infer_forward_layer(ctx, layer, pos);
            }
            infer_complete_deferred();
            pos++;

            infer_final_norm(ctx);
            infer_lm_head(ctx);
        } /* @autoreleasepool */ }

        gen_done:;

        free(logits_copy);

        // Save session position for KV cache continuity
        // (pos now includes all prefill + generated tokens)
        if (req.cache && req.session_id[0] &&
            !(finish_reason && strcmp(finish_reason, "abort") == 0)) {
            session_pos = pos;
            fprintf(stderr, "[worker] %s session_pos=%d (session=%s)\n",
                    req.request_id, session_pos, active_session);
        }

        double gen_ms = now_ms() - t_gen;
        double tok_per_sec = gen_count > 0 ? gen_count * 1000.0 / gen_ms : 0.0;
        fprintf(stderr, "[worker] %s generated=%d tokens in %.0fms (%.2f tok/s)%s\n",
                req.request_id, gen_count, gen_ms, tok_per_sec,
                (finish_reason && strcmp(finish_reason, "abort") == 0) ? " [ABORTED]" : "");

        // On abort: invalidate session and cleanup
        if (finish_reason && strcmp(finish_reason, "abort") == 0) {
            active_session[0] = '\0';
            session_pos = 0;
            free(response_buf);
            free_completion_request(&req);
            close(client_fd);
            continue;
        }

        // Check for tool calls in response
        ParsedToolCall tool_calls[8];
        int num_tool_calls = 0;
        if (strstr(response_buf, "<tool_call>")) {
            num_tool_calls = parse_tool_calls(response_buf, tool_calls, 8);
            if (num_tool_calls > 0) {
                finish_reason = "tool_calls";
            }
        }

        // Send response
        if (req.stream) {
            // Flush any remaining stop detector buffer through UTF-8 buffer
            if (sd.buf_len > 0) {
                char u8out[4096];
                const char *safe = utf8_stream_push(&u8buf, sd.buffer, sd.buf_len, u8out, sizeof(u8out));
                if (safe[0]) sse_send_delta(client_fd, &req, safe);
            }
            // Flush any remaining incomplete UTF-8 bytes
            {
                char u8out[8];
                const char *safe = utf8_stream_flush(&u8buf, u8out, sizeof(u8out));
                if (safe[0]) sse_send_delta(client_fd, &req, safe);
            }
            // Send tool call deltas before finish (if any)
            if (num_tool_calls > 0) {
                sse_send_tool_calls(client_fd, &req, tool_calls, num_tool_calls);
            }
            sse_send_finish(client_fd, &req, finish_reason, prompt_tokens, gen_count, cached_tokens, prefill_tps, tok_per_sec);
            sse_send_done(client_fd);
        } else {
            // Extract content before tool calls (if any)
            char *content = response_buf;
            if (num_tool_calls > 0) {
                char *tc_start = strstr(response_buf, "<tool_call>");
                if (tc_start && tc_start > response_buf) {
                    *tc_start = '\0';
                    // Trim trailing whitespace
                    int clen = (int)strlen(content);
                    while (clen > 0 && (content[clen-1] == '\n' || content[clen-1] == ' '))
                        content[--clen] = '\0';
                }
            }

            char *resp = build_completion_response(&req, content, tool_calls, num_tool_calls,
                                                     finish_reason, prompt_tokens, gen_count,
                                                     cached_tokens, prefill_tps, tok_per_sec);
            send_json_response(client_fd, 200, resp);
            free(resp);
        }

        free(response_buf);
        free_completion_request(&req);
        close(client_fd);
        } // @autoreleasepool
    }

    free(token_counts);
    return NULL;
}

// ============================================================================
// HTTP accept and routing
// ============================================================================

static void handle_models(int client_fd) {
    // Derive model name from config
    char model_name[256];
    const char *mp = cfg.model_path;
    // Use last path component
    const char *last = strrchr(mp, '/');
    if (last) last++; else last = mp;
    strlcpy(model_name, last, sizeof(model_name));

    struct timeval tv;
    gettimeofday(&tv, NULL);

    char body[1024];
    snprintf(body, sizeof(body),
        "{\"object\":\"list\",\"data\":[{"
        "\"id\":\"%s\","
        "\"object\":\"model\","
        "\"created\":%ld,"
        "\"owned_by\":\"flash-moe\""
        "}]}",
        model_name, tv.tv_sec);

    send_json_response(client_fd, 200, body);
}

static void handle_health(int client_fd) {
    char body[256];
    snprintf(body, sizeof(body),
        "{\"status\":\"ok\",\"model\":\"%s\","
        "\"vocab_size\":%d,\"hidden_dim\":%d,\"num_layers\":%d}",
        cfg.model_path, cfg.vocab_size, cfg.hidden_dim, cfg.num_layers);
    send_json_response(client_fd, 200, body);
}

// ============================================================================
// System prompt pre-caching (same approach as serve_loop in infer.m)
// ============================================================================

static InferStateSnapshot *precache_system_prompt(InferContext *ctx) {
    fprintf(stderr, "[server] Pre-caching system prompt...\n");

    extern char *load_system_prompt(void);
    char *sys = load_system_prompt();

    size_t total = 100 + strlen(sys);
    char *prompt = malloc(total);
    snprintf(prompt, total,
        "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n",
        sys);

    PromptTokens *pt = infer_encode_text(prompt);
    free(prompt);
    free(sys);
    if (!pt || pt->count == 0) {
        if (pt) infer_free_tokens(pt);
        fprintf(stderr, "WARNING: Failed to tokenize system prompt\n");
        return NULL;
    }

    // Reset state — only zeros the used portion of KV caches (len=0 after calloc,
    // so this is effectively a no-op on fresh context, avoiding 5+ GB memset)
    infer_reset_state(ctx);

    // Prefill (fast: skip routed experts for intermediate tokens)
    int pos = 0;
    for (int i = 0; i < pt->count - 1; i++) {
        infer_embed_token(ctx, pt->ids[i]);
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            infer_forward_layer(ctx, layer, pos);
        }
        infer_discard_deferred_nowait();
        pos++;
    }
    // Last token: full forward (need accurate hidden state)
    infer_embed_token(ctx, pt->ids[pt->count - 1]);
    for (int layer = 0; layer < cfg.num_layers; layer++) {
        infer_forward_layer(ctx, layer, pos);
    }
    infer_complete_deferred();
    pos++;

    infer_sync_delta_state(ctx);

    fprintf(stderr, "[server] System prompt cached: %d tokens prefilled\n", pos);
    infer_free_tokens(pt);

    return infer_snapshot_state(ctx, pos);
}

// ============================================================================
// Main
// ============================================================================

static void print_usage(void) {
    printf("Usage: server [options]\n");
    printf("  --model PATH       Model directory (or set FLASH_MOE_MODEL)\n");
    printf("  --port PORT        HTTP port (default: 8080)\n");
    printf("  --host HOST        Bind host (default: 0.0.0.0)\n");
    printf("  --weights PATH     model_weights.bin path\n");
    printf("  --manifest PATH    model_weights.json path\n");
    printf("  --vocab PATH       vocab.bin path\n");
    printf("  --k N              Active experts per layer (default: from config)\n");
    printf("  --tiered           Force tiered quantization (hot=4-bit, cold=2-bit)\n");
    printf("  --2bit             Force 2-bit expert quantization\n");
    printf("  --fp8              Use FP8 E4M3 KV cache (4x memory reduction)\n");
    printf("  --fused-attn       Enable fused online softmax attention (experimental)\n");
    printf("  --fp16             Use FP16 accumulation in dequant kernels (experimental)\n");
    printf("  --fused-expert     Enable fused gate+up+SwiGLU expert kernel\n");
    printf("  --no-fused-expert  Disable fused gate+up+SwiGLU expert kernel\n");
    printf("  --no-cmd-merge     Disable CMD1+CMD2 merge for linear attention\n");
    printf("  --expert-prefetch  Enable cross-layer expert prefetch\n");
    printf("  --help             This message\n");
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *model_path = getenv("FLASH_MOE_MODEL");
        const char *weights_path = NULL;
        const char *manifest_path = NULL;
        const char *vocab_path = NULL;
        int port = DEFAULT_PORT;
        int K = 0;  // 0 = auto from config
        int use_tiered = 0;
        int use_2bit = 0;

        static struct option long_options[] = {
            {"model",     required_argument, 0, 'm'},
            {"port",      required_argument, 0, 'p'},
            {"weights",   required_argument, 0, 'w'},
            {"manifest",  required_argument, 0, 'j'},
            {"vocab",     required_argument, 0, 'v'},
            {"k",         required_argument, 0, 'k'},
            {"tiered",    no_argument,       0, 't'},
            {"2bit",      no_argument,       0, '2'},
            {"fp8",           no_argument,   0, 1001},
            {"fused-attn",    no_argument,   0, 1002},
            {"fp16",          no_argument,   0, 1003},
            {"fused-expert",  no_argument,   0, 1007},
            {"no-fused-expert", no_argument, 0, 1004},
            {"cmd-merge",     no_argument,   0, 1008},
            {"no-cmd-merge",  no_argument,   0, 1005},
            {"expert-prefetch", no_argument, 0, 1006},
            {"help",      no_argument,       0, 'h'},
            {0, 0, 0, 0}
        };

        int opt;
        while ((opt = getopt_long(argc, argv, "m:p:w:j:v:k:t2h", long_options, NULL)) != -1) {
            switch (opt) {
                case 'm': model_path = optarg; break;
                case 'p': port = atoi(optarg); break;
                case 'w': weights_path = optarg; break;
                case 'j': manifest_path = optarg; break;
                case 'v': vocab_path = optarg; break;
                case 'k': K = atoi(optarg); break;
                case 't': use_tiered = 1; break;
                case '2': use_2bit = 1; break;
                case 1001: infer_set_fp8_kv(1); break;
                case 1002: infer_set_fused_attention(1); break;
                case 1003: infer_set_fp16_accum(1); break;
                case 1004: infer_set_fused_expert(0); break;
                case 1005: infer_set_cmd_merge(0); break;
                case 1006: infer_set_expert_prefetch(1); break;
                case 1007: infer_set_fused_expert(1); break;
                case 1008: infer_set_cmd_merge(1); break;
                case 'h': print_usage(); return 0;
                default: print_usage(); return 1;
            }
        }

        if (!model_path) {
            fprintf(stderr, "ERROR: No model path. Use --model or set FLASH_MOE_MODEL\n");
            print_usage();
            return 1;
        }

        printf("=== Flash-MoE OpenAI API Server ===\n");
        printf("Model:  %s\n", model_path);
        printf("Port:   %d\n", port);

        // Initialize inference engine
        InferContext *ctx = infer_init(model_path, weights_path, manifest_path, vocab_path,
                                       K > 0 ? K : 8, use_tiered, use_2bit);
        if (!ctx) {
            fprintf(stderr, "ERROR: Failed to initialize inference engine\n");
            return 1;
        }

        // Use K from config if not specified
        if (K == 0) K = cfg.num_experts_per_tok;
        ctx->K = K;

        // Pre-cache system prompt (like serve_loop in infer.m)
        InferStateSnapshot *sys_snap = precache_system_prompt(ctx);
        int sys_pos = sys_snap ? sys_snap->pos : 0;

        // Create request queue
        RequestQueue queue;
        queue_init(&queue);

        // Start inference worker thread
        InferWorkerArgs worker_args = {
            .ctx = ctx,
            .queue = &queue,
            .sys_snapshot = sys_snap,
            .sys_pos = sys_pos
        };
        pthread_t worker_tid;
        pthread_attr_t worker_attr;
        pthread_attr_init(&worker_attr);
        pthread_attr_setstacksize(&worker_attr, 8 * 1024 * 1024);  // 8MB stack
        pthread_create(&worker_tid, &worker_attr, inference_worker, &worker_args);
        pthread_attr_destroy(&worker_attr);

        // Ignore SIGPIPE
        signal(SIGPIPE, SIG_IGN);

        // Create server socket
        int server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd < 0) { perror("socket"); return 1; }

        int opt_val = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt_val, sizeof(opt_val));

        struct sockaddr_in addr = {0};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port);

        if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            perror("bind"); return 1;
        }
        if (listen(server_fd, 32) < 0) {
            perror("listen"); return 1;
        }

        printf("\n[server] Listening on http://0.0.0.0:%d\n", port);
        printf("[server] POST /v1/chat/completions\n");
        printf("[server] GET  /v1/models\n");
        printf("[server] GET  /health\n\n");
        fflush(stdout);

        // Accept loop
        for (;;) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
            if (client_fd < 0) { perror("accept"); continue; }

            // Read HTTP request
            char *reqbuf = malloc(MAX_REQUEST_SIZE);
            int reqlen = read_http_request(client_fd, reqbuf, MAX_REQUEST_SIZE);
            if (reqlen <= 0) { free(reqbuf); close(client_fd); continue; }

            // Parse method and path
            char method[16] = {0}, path[256] = {0};
            sscanf(reqbuf, "%15s %255s", method, path);

            // OPTIONS (CORS preflight)
            if (strcmp(method, "OPTIONS") == 0) {
                char resp[256];
                snprintf(resp, sizeof(resp),
                    "HTTP/1.1 204 No Content\r\n%sAccess-Control-Max-Age: 86400\r\n\r\n",
                    CORS_HEADERS);
                http_write_str(client_fd, resp);
                free(reqbuf); close(client_fd);
                continue;
            }

            // GET /health
            if (strcmp(method, "GET") == 0 && strcmp(path, "/health") == 0) {
                handle_health(client_fd);
                free(reqbuf); close(client_fd);
                continue;
            }

            // GET /v1/models
            if (strcmp(method, "GET") == 0 &&
                (strcmp(path, "/v1/models") == 0 || strcmp(path, "/models") == 0)) {
                handle_models(client_fd);
                free(reqbuf); close(client_fd);
                continue;
            }

            // POST /v1/chat/completions
            if (strcmp(method, "POST") == 0 &&
                (strcmp(path, "/v1/chat/completions") == 0 ||
                 strcmp(path, "/chat/completions") == 0)) {
                // Find body
                char *body = strstr(reqbuf, "\r\n\r\n");
                if (!body) {
                    send_error(client_fd, 400, "No request body", "invalid_request_error");
                    free(reqbuf); close(client_fd);
                    continue;
                }
                body += 4;
                int body_len = reqlen - (int)(body - reqbuf);

                // Copy body for the queue (reqbuf will be freed)
                char *body_copy = malloc(body_len + 1);
                memcpy(body_copy, body, body_len);
                body_copy[body_len] = '\0';

                // Enqueue for inference worker
                queue_push(&queue, client_fd, body_copy, body_len);
                free(reqbuf);
                // Don't close client_fd — it's owned by the worker now
                continue;
            }

            // Unknown endpoint
            send_error(client_fd, 404,
                "Not found. Available: POST /v1/chat/completions, GET /v1/models, GET /health",
                "not_found");
            free(reqbuf); close(client_fd);
        }

        // Never reached
        return 0;
    }
}
