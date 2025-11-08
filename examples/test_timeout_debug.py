"""Debug script to test timeout behavior."""

import openai
import time

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

print("Testing timeout with very short duration...")
print("=" * 80)

try:
    start = time.time()
    response = client.completions.create(
        model="facebook/opt-125m",
        prompt="Write a long story:",
        max_tokens=200,
        temperature=0.8,
        extra_body={"max_execution_time": 0.1},  # Very short - 100ms timeout
        timeout=10.0,  # Client timeout of 10 seconds
    )
    elapsed = time.time() - start

    print(f"Request completed in {elapsed:.2f}s")
    print(f"Response: {response.choices[0].text[:100] if response.choices[0].text else '(empty)'}...")
    print(f"Finish reason: {response.choices[0].finish_reason}")
    print(f"Tokens in response: {len(response.choices[0].text.split()) if response.choices[0].text else 0}")

except Exception as e:
    elapsed = time.time() - start
    print(f"Error after {elapsed:.2f}s: {type(e).__name__}: {e}")

print("=" * 80)
