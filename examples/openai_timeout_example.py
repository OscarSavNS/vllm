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

print("Example 1: Completion with short timeout to trigger timeout")
print("="*80)

# Text completion with very short timeout to demonstrate timeout behavior
response = client.completions.create(
    model="facebook/opt-125m",
    prompt="Write a very long story about space exploration:",
    max_tokens=500,  # Request many tokens
    temperature=0.8,
    extra_body={"max_execution_time": 0.5},  # Very short timeout (0.5s) to trigger timeout
)

print(f"Response: {response.choices[0].text[:200]}...")  # Show first 200 chars
print(f"Finish reason: {response.choices[0].finish_reason}")

if response.choices[0].finish_reason == "timeout":
    print("✓ This request timed out as expected!")
    print("  Note: Partial results were still returned.")

print("\n" + "="*80)
print("Example 2: Completion with sufficient timeout (no timeout expected)")
print("="*80)

# Text completion with reasonable timeout - should complete normally
response = client.completions.create(
    model="facebook/opt-125m",
    prompt="The meaning of life is",
    max_tokens=50,
    temperature=0.7,
    extra_body={"max_execution_time": 30.0},  # Generous timeout
)

print(f"Response: {response.choices[0].text}")
print(f"Finish reason: {response.choices[0].finish_reason}")

if response.choices[0].finish_reason != "timeout":
    print("✓ Request completed normally within timeout window.")

print("\n" + "="*80)
print("Example 3: Streaming completion with timeout")
print("="*80)

# Streaming with timeout
stream = client.completions.create(
    model="facebook/opt-125m",
    prompt="Count from 1 to 100:",
    max_tokens=500,
    temperature=0.8,
    stream=True,
    extra_body={"max_execution_time": 1.0},  # Short timeout to demonstrate streaming timeout
)

print("Streaming response:")
for chunk in stream:
    if chunk.choices[0].text:
        print(chunk.choices[0].text, end="", flush=True)

    # Check finish reason on the last chunk
    if chunk.choices[0].finish_reason:
        print(f"\n\nFinish reason: {chunk.choices[0].finish_reason}")
        if chunk.choices[0].finish_reason == "timeout":
            print("✓ Stream was terminated due to timeout as expected!")
            print("  Note: Partial streaming results were still returned.")

print("\n" + "="*80)
print("\nUsage notes:")
print("- max_execution_time is in seconds")
print("- Pass via extra_body parameter when using OpenAI client")
print("- Includes time spent waiting in queue")
print("- Works with both streaming and non-streaming requests")
print("- Returns finish_reason='timeout' when triggered")
print("- Set to None (or omit) for no timeout")
