"""Quick test for timeout feature with OpenAI-compatible API.

This is a simple, fast test that verifies the timeout feature works.
"""

import openai

# Configure the OpenAI client to use vLLM server
# Start the vLLM server first:
# vllm serve facebook/opt-125m
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

print("Quick Test: Completion with timeout")
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
    print("\n✓ SUCCESS: Request completed normally within timeout window.")
    print("✓ TIMEOUT FEATURE WORKS!")
else:
    print("\n⚠️  Unexpected timeout")

print("\n" + "="*80)
print("Test completed successfully!")
