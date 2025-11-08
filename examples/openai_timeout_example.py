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

print("Example 1: Completion with generous timeout (should complete normally)")
print("="*80)

# Text completion with generous timeout - should complete successfully
response = client.completions.create(
    model="facebook/opt-125m",
    prompt="The capital of France is",
    max_tokens=50,
    temperature=0.7,
    extra_body={"max_execution_time": 30.0},  # Generous 30-second timeout
)

print(f"Response: {response.choices[0].text}")
print(f"Finish reason: {response.choices[0].finish_reason}")

if response.choices[0].finish_reason != "timeout":
    print("✓ Request completed normally within timeout window!")

print("\n" + "="*80)
print("Example 2: Completion with tight timeout (may timeout)")
print("="*80)

# Text completion with short timeout to demonstrate timeout behavior
# Note: Depending on system load, this may or may not timeout
response = client.completions.create(
    model="facebook/opt-125m",
    prompt="Write a long story:",
    max_tokens=200,
    temperature=0.8,
    extra_body={"max_execution_time": 0.5},  # 0.5 second timeout
)

print(f"Response: {response.choices[0].text}...")  # Show first 150 chars
print(f"Finish reason: {response.choices[0].finish_reason}")

if response.choices[0].finish_reason == "timeout":
    print("⚠️  This request timed out!")
    print("   Note: Partial results were still returned (graceful timeout).")
else:
    print("✓ Request completed within timeout window.")

print("\n" + "="*80)
print("Example 3: Streaming completion")
print("="*80)

# Streaming with generous timeout
stream = client.completions.create(
    model="facebook/opt-125m",
    prompt="Count from 1 to 10:",
    max_tokens=100,
    temperature=0.7,
    stream=True,
    extra_body={"max_execution_time": 30.0},  # Generous timeout
)

print("Streaming response:")
for chunk in stream:
    if chunk.choices[0].text:
        print(chunk.choices[0].text, end="", flush=True)

    # Check finish reason on the last chunk
    if chunk.choices[0].finish_reason:
        print(f"\n\nFinish reason: {chunk.choices[0].finish_reason}")
        if chunk.choices[0].finish_reason == "timeout":
            print("⚠️  Stream was terminated due to timeout!")
        else:
            print("✓ Stream completed normally!")

print("\n" + "="*80)
print("\nUsage notes:")
print("- max_execution_time is in seconds")
print("- Pass via extra_body parameter when using OpenAI client")
print("- Includes time spent waiting in queue")
print("- Works with both streaming and non-streaming requests")
print("- Returns finish_reason='timeout' when triggered")
print("- Set to None (or omit) for no timeout")
