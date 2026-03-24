#!/usr/bin/env python3
"""Quick test for the Flash-MoE OpenAI-compatible API server."""

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)

# 3. Streaming completion
print("\n=== Streaming ===")
stream = client.chat.completions.create(
    model="flash-moe",
    messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "<think>\n\n</think>\n"},
    ],
    # max_tokens=50,
    temperature=0.1,
    stream=True,
)
print("  ", end="", flush=True)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
print(f"\n  finish: {chunk.choices[0].finish_reason}")
