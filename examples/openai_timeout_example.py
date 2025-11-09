"""Example demonstrating timeout feature with OpenAI-compatible API.

Start the vLLM server first:
    vllm serve facebook/opt-125m

Then run this example:
    python examples/openai_timeout_example.py
"""

import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

# Example 1: Completion with timeout
print("Example 1: Completion with 30-second timeout")
print("=" * 80)

response = client.completions.create(
    model="facebook/opt-125m",
    prompt="The capital of France is",
    max_tokens=50,
    temperature=0.7,
    extra_body={"max_execution_time": 30.0},
)

print(f"Response: {response.choices[0].text}")
print(f"Finish reason: {response.choices[0].finish_reason}")

if response.choices[0].finish_reason != "timeout":
    print("✓ Request completed normally within timeout window")

# Example 2: Streaming completion with timeout
print("\n" + "=" * 80)
print("Example 2: Streaming completion with 30-second timeout")
print("=" * 80)

stream = client.completions.create(
    model="facebook/opt-125m",
    prompt="Count from 1 to 10:",
    max_tokens=100,
    temperature=0.7,
    stream=True,
    extra_body={"max_execution_time": 30.0},
)

print("Streaming response: ", end="", flush=True)
for chunk in stream:
    if chunk.choices[0].text:
        print(chunk.choices[0].text, end="", flush=True)

    if chunk.choices[0].finish_reason:
        print(f"\n\nFinish reason: {chunk.choices[0].finish_reason}")
        if chunk.choices[0].finish_reason == "timeout":
            print("⚠️  Stream timed out (partial results returned)")
        else:
            print("✓ Stream completed normally")

# Example 3: Chat completion with timeout
print("\n" + "=" * 80)
print("Example 3: Chat completion with 30-second timeout")
print("=" * 80)

response = client.chat.completions.create(
    model="facebook/opt-125m",
    messages=[
        {"role": "user", "content": "What is the meaning of life?"}
    ],
    max_tokens=50,
    temperature=0.7,
    extra_body={"max_execution_time": 30.0},
)

print(f"Response: {response.choices[0].message.content}")
print(f"Finish reason: {response.choices[0].finish_reason}")

print("\n" + "=" * 80)
print("\nUsage notes:")
print("- Pass timeout via extra_body={'max_execution_time': seconds}")
print("- Works with completions, chat, and streaming endpoints")
print("- Returns partial results with finish_reason='timeout'")
print("- Omit parameter for no timeout (default)")
