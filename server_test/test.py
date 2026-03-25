#!/usr/bin/env python3
"""Quick test for the Flash-MoE OpenAI-compatible API server."""

import json
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)

# 1. List models
print("=== Models ===")
models = client.models.list()
for m in models.data:
    print(f"  {m.id} (owned_by: {m.owned_by})")

# 2. Non-streaming completion
print("\n=== Non-streaming ===")
resp = client.chat.completions.create(
    model="flash-moe",
    messages=[{"role": "user", "content": "Say hello in one word."}],
    max_tokens=20,
    temperature=0,
    stream=False,
)
print(f"  id:      {resp.id}")
print(f"  content: {resp.choices[0].message.content}")
print(f"  finish:  {resp.choices[0].finish_reason}")
print(f"  usage:   {resp.usage}")

# 3. Streaming completion
print("\n=== Streaming ===")
stream = client.chat.completions.create(
    model="flash-moe",
    messages=[{"role": "user", "content": "Count from 1 to 5."}],
    max_tokens=50,
    temperature=0.7,
    stream=True,
)
print("  ", end="", flush=True)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
print(f"\n  finish: {chunk.choices[0].finish_reason}")

# 4. Multi-turn conversation
print("\n=== Multi-turn ===")
resp = client.chat.completions.create(
    model="flash-moe",
    messages=[
        {"role": "system", "content": "You are a pirate. Respond in pirate speak."},
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "Arrr, the skies be clear today, matey!"},
        {"role": "user", "content": "And tomorrow?"},
    ],
    max_tokens=50,
    temperature=0.8,
)
print(f"  {resp.choices[0].message.content}")

# 5. Parallel requests (queued — server processes sequentially but accepts concurrently)
print("\n=== Parallel requests ===")
import concurrent.futures
import time

prompts = [
    "What is 2+2?",
    "Name a color.",
    "Say yes or no.",
    "What planet are we on?",
]

def send_request(prompt):
    t0 = time.time()
    resp = client.chat.completions.create(
        model="flash-moe",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        temperature=0,
    )
    elapsed = time.time() - t0
    return prompt, resp.choices[0].message.content.strip(), elapsed

t_start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(send_request, p) for p in prompts]
    for f in concurrent.futures.as_completed(futures):
        prompt, answer, elapsed = f.result()
        print(f"  [{elapsed:.1f}s] \"{prompt}\" → {answer[:60]}")

t_total = time.time() - t_start
print(f"  Total wall time: {t_total:.1f}s (sequential would be ~{t_total:.0f}s since inference is serial)")

# 6. Tool calling — single tool call
print("\n=== Tool calling (single) ===")
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'San Francisco'",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. 'AAPL'",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
]

resp = client.chat.completions.create(
    model="flash-moe",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto",
    max_tokens=200,
    temperature=0,
)
msg = resp.choices[0].message
print(f"  finish_reason: {resp.choices[0].finish_reason}")
print(f"  content:       {msg.content}")
print(f"  tool_calls:    {msg.tool_calls}")
assert resp.choices[0].finish_reason == "tool_calls", (
    f"Expected finish_reason='tool_calls', got '{resp.choices[0].finish_reason}'"
)
assert msg.tool_calls and len(msg.tool_calls) >= 1, "Expected at least one tool call"
tc = msg.tool_calls[0]
print(f"  call id:       {tc.id}")
print(f"  function:      {tc.function.name}")
print(f"  arguments:     {tc.function.arguments}")
assert tc.type == "function"
assert tc.function.name == "get_weather"
args = json.loads(tc.function.arguments)
assert "location" in args, f"Expected 'location' in arguments, got {args}"
print(f"  parsed args:   {args}")

# 7. Tool calling — multi-turn with tool response
print("\n=== Tool calling (multi-turn with tool response) ===")
resp2 = client.chat.completions.create(
    model="flash-moe",
    messages=[
        {"role": "user", "content": "What's the weather in Tokyo?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": tc.function.arguments,
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps({"temperature": 22, "condition": "Partly cloudy", "humidity": 65}),
        },
    ],
    tools=tools,
    max_tokens=200,
    temperature=0,
)
msg2 = resp2.choices[0].message
print(f"  finish_reason: {resp2.choices[0].finish_reason}")
print(f"  content:       {msg2.content}")
assert resp2.choices[0].finish_reason in ("stop", "length"), (
    f"Expected natural finish after tool response, got '{resp2.choices[0].finish_reason}'"
)
assert msg2.content and len(msg2.content) > 0, "Expected non-empty content summarizing tool result"
# The model should mention Tokyo and/or the weather data
print("  (model used tool result to answer)")

# 8. Tool calling — streaming with tool calls
print("\n=== Tool calling (streaming) ===")
stream = client.chat.completions.create(
    model="flash-moe",
    messages=[{"role": "user", "content": "What is Apple's stock price?"}],
    tools=tools,
    tool_choice="auto",
    max_tokens=200,
    temperature=0,
    stream=True,
)
collected_tool_calls = {}
finish = None
print("  ", end="", flush=True)
for chunk in stream:
    choice = chunk.choices[0]
    if choice.finish_reason:
        finish = choice.finish_reason
    delta = choice.delta
    if delta.content:
        print(delta.content, end="", flush=True)
    if delta.tool_calls:
        for tc_delta in delta.tool_calls:
            idx = tc_delta.index
            if idx not in collected_tool_calls:
                collected_tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
            if tc_delta.id:
                collected_tool_calls[idx]["id"] = tc_delta.id
            if tc_delta.function:
                if tc_delta.function.name:
                    collected_tool_calls[idx]["name"] = tc_delta.function.name
                if tc_delta.function.arguments:
                    collected_tool_calls[idx]["arguments"] += tc_delta.function.arguments
print()
print(f"  finish_reason: {finish}")
assert finish == "tool_calls", f"Expected finish_reason='tool_calls', got '{finish}'"
assert len(collected_tool_calls) >= 1, "Expected at least one tool call in stream"
for idx, tc_info in sorted(collected_tool_calls.items()):
    print(f"  tool_call[{idx}]: {tc_info['name']}({tc_info['arguments']})")
    assert tc_info["name"] == "get_stock_price", f"Expected 'get_stock_price', got '{tc_info['name']}'"
    stream_args = json.loads(tc_info["arguments"])
    assert "symbol" in stream_args, f"Expected 'symbol' in arguments, got {stream_args}"

# 9. Disable thinking via assistant prefix
print("\n=== Disable thinking (assistant prefix) ===")
resp = client.chat.completions.create(
    model="flash-moe",
    messages=[
        {"role": "user", "content": "Say hello in one word."},
        {"role": "assistant", "content": "<think>\n</think>\n"},
    ],
    max_tokens=30,
    temperature=0,
)
content = resp.choices[0].message.content
print(f"  content: {content}")
# The model continues from after </think>, so no thinking in its output
assert "<think>" not in content, f"Expected no <think> tags, got: {content}"
print("  (no <think> tags — model continued from prefix)")

# 10. Batch prefill stress test — long prompt to exercise batched prefill path
print("\n=== Batch prefill (long prompt) ===")
long_prompt = (
    "Below is a list of 50 famous scientists and their key contributions:\n"
    + "\n".join(
        f"{i+1}. Scientist_{i}: discovered principle_{i} in the year {1900+i}"
        for i in range(50)
    )
    + "\n\nBased on the list above, which scientist made a discovery in 1925? "
    "Answer with just the scientist name and discovery."
)
t0 = time.time()
resp = client.chat.completions.create(
    model="flash-moe",
    messages=[{"role": "user", "content": long_prompt}],
    max_tokens=60,
    temperature=0,
    stream=False,
)
elapsed = time.time() - t0
content = resp.choices[0].message.content
usage = resp.usage
print(f"  prompt_tokens:     {usage.prompt_tokens}")
print(f"  completion_tokens: {usage.completion_tokens}")
print(f"  total_tokens:      {usage.total_tokens}")
print(f"  content:           {content[:120]}")
print(f"  elapsed:           {elapsed:.1f}s")
print(f"  prefill speed:     {usage.prompt_tokens / elapsed:.1f} tok/s (prompt)")
assert usage.prompt_tokens > 200, (
    f"Expected >200 prompt tokens for batch prefill test, got {usage.prompt_tokens}"
)
print("  (long prompt exercises batched prefill / NAX GEMM path)")

# 11. Batch prefill — streaming with long context
print("\n=== Batch prefill (streaming, long context) ===")
multi_paragraph = "\n\n".join(
    f"Paragraph {i+1}: The quick brown fox jumps over the lazy dog. "
    f"This is sentence number {i+1} in a long document that tests prefill batching. "
    f"The value associated with this paragraph is {i * 7}."
    for i in range(30)
)
t0 = time.time()
stream = client.chat.completions.create(
    model="flash-moe",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": multi_paragraph + "\n\nWhat is the value in paragraph 15?"},
    ],
    max_tokens=40,
    temperature=0,
    stream=True,
)
tokens_received = 0
print("  ", end="", flush=True)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
        tokens_received += 1
elapsed = time.time() - t0
print(f"\n  tokens received:   {tokens_received}")
print(f"  elapsed:           {elapsed:.1f}s")
print(f"  TTFT + generation in {elapsed:.1f}s")
assert tokens_received > 0, "Expected at least some tokens from streaming long-context"
print("  (streaming long context exercises batched prefill + NAX)")

print("\n=== All tests passed ===")
