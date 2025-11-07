"""Example demonstrating timeout feature with OpenAI-compatible API.

This example shows how to use the max_execution_time parameter
with vLLM's OpenAI-compatible API endpoints.
"""

import openai

# Configure the OpenAI client to use vLLM server
# Start the vLLM server first:
# vllm serve facebook/opt-125m
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

print("Example 1: Chat completion with timeout")
print("="*80)

# Chat completion with 10-second timeout
response = client.chat.completions.create(
    model="facebook/opt-125m",
    messages=[
        {"role": "user", "content": "Write a very long story about space exploration."}
    ],
    max_tokens=500,
    temperature=0.8,
    extra_body={"max_execution_time": 10.0},  # 10-second timeout via extra_body
)

print(f"Response: {response.choices[0].message.content}")
print(f"Finish reason: {response.choices[0].finish_reason}")

if response.choices[0].finish_reason == "timeout":
    print("⚠️  This request timed out after 10 seconds!")

print("\n" + "="*80)
print("Example 2: Completion with timeout")
print("="*80)

# Text completion with 5-second timeout
response = client.completions.create(
    model="facebook/opt-125m",
    prompt="The meaning of life is",
    max_tokens=200,
    temperature=0.7,
    extra_body={"max_execution_time": 5.0},  # 5-second timeout via extra_body
)

print(f"Response: {response.choices[0].text}")
print(f"Finish reason: {response.choices[0].finish_reason}")

if response.choices[0].finish_reason == "timeout":
    print("⚠️  This request timed out after 5 seconds!")

print("\n" + "="*80)
print("Example 3: Streaming chat completion with timeout")
print("="*80)

# Streaming with timeout
stream = client.chat.completions.create(
    model="facebook/opt-125m",
    messages=[
        {"role": "user", "content": "Count to 100."}
    ],
    max_tokens=500,
    temperature=0.8,
    stream=True,
    extra_body={"max_execution_time": 3.0},  # 3-second timeout via extra_body
)

print("Streaming response:")
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

    # Check finish reason on the last chunk
    if chunk.choices[0].finish_reason:
        print(f"\n\nFinish reason: {chunk.choices[0].finish_reason}")
        if chunk.choices[0].finish_reason == "timeout":
            print("⚠️  Stream was terminated due to timeout!")

print("\n" + "="*80)
print("\nUsage notes:")
print("- max_execution_time is in seconds")
print("- Pass via extra_body parameter when using OpenAI client")
print("- Includes time spent waiting in queue")
print("- Works with both streaming and non-streaming requests")
print("- Returns finish_reason='timeout' when triggered")
print("- Set to None (or omit) for no timeout")
