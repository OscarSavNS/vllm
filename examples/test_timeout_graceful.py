"""Test script to verify timeout returns partial results gracefully.

This script demonstrates that when a request times out, it returns
whatever tokens have been generated up to that point, rather than
discarding them.
"""

import time

from vllm import LLM, SamplingParams

# Initialize a small model for faster testing
llm = LLM(model="facebook/opt-125m", max_model_len=512)

print("=" * 80)
print("Test: Timeout returns partial results")
print("=" * 80)

# Create a prompt that will generate many tokens
prompt = "Write a very long story about a dragon: Once upon a time"

# Set a very short timeout (2 seconds) to force a timeout
sampling_params = SamplingParams(
    max_tokens=500,  # Request many tokens
    max_execution_time=2.0,  # But timeout after 2 seconds
    temperature=0.8,
)

print(f"\nPrompt: {prompt}")
print(f"Max tokens: {sampling_params.max_tokens}")
print(f"Timeout: {sampling_params.max_execution_time}s")
print("\nGenerating...")

start_time = time.time()
outputs = llm.generate([prompt], sampling_params)
elapsed_time = time.time() - start_time

output = outputs[0]
generated_text = output.outputs[0].text
finish_reason = output.outputs[0].finish_reason
num_tokens = len(output.outputs[0].token_ids)

print(f"\nElapsed time: {elapsed_time:.2f}s")
print(f"Finish reason: {finish_reason}")
print(f"Tokens generated: {num_tokens}")
print(f"\nGenerated text:\n{generated_text}")

# Verify timeout behavior
print("\n" + "=" * 80)
print("Verification:")
print("=" * 80)

if finish_reason == "timeout":
    print("✓ Request correctly timed out")
    if num_tokens > 0:
        print(f"✓ Partial results returned: {num_tokens} tokens")
        print("✓ GRACEFUL TIMEOUT WORKS!")
    else:
        print("✗ No tokens returned - partial results lost!")
else:
    print(f"✗ Unexpected finish reason: {finish_reason}")
    print("  (Expected 'timeout')")

if elapsed_time < sampling_params.max_execution_time + 1.0:
    print(f"✓ Timeout enforced (within {elapsed_time:.2f}s)")
else:
    print(f"✗ Timeout not enforced (took {elapsed_time:.2f}s)")

print("\n" + "=" * 80)
print("Test: No timeout (normal completion)")
print("=" * 80)

# Now test without timeout for comparison
sampling_params_no_timeout = SamplingParams(
    max_tokens=50,  # Fewer tokens
    max_execution_time=None,  # No timeout
    temperature=0.8,
)

print(f"\nGenerating with no timeout (max_tokens={sampling_params_no_timeout.max_tokens})...")
outputs = llm.generate([prompt], sampling_params_no_timeout)

output = outputs[0]
finish_reason = output.outputs[0].finish_reason
num_tokens = len(output.outputs[0].token_ids)

print(f"Finish reason: {finish_reason}")
print(f"Tokens generated: {num_tokens}")

if finish_reason != "timeout":
    print("✓ No timeout occurred as expected")
else:
    print("✗ Unexpected timeout!")
