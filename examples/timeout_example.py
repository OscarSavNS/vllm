"""Example demonstrating the request timeout feature in vLLM.

This example shows how to use the max_execution_time parameter to set
timeouts on generation requests.
"""

from vllm import LLM, SamplingParams


def main():
    # Initialize the LLM
    llm = LLM(model="facebook/opt-125m")

    # Example 1: Request with a timeout
    # This request will be terminated after 5 seconds if not completed
    prompts = [
        "What is the capital of France?",
        "Write a very long story about a dragon.",
    ]

    # Set a 5-second timeout
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=1000,
        max_execution_time=5.0,  # Timeout after 5 seconds
    )

    print("Running generation with 5-second timeout...")
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        finish_reason = output.outputs[0].finish_reason

        print(f"\nPrompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}")
        print(f"Finish reason: {finish_reason}")

        if finish_reason == "timeout":
            print("⚠️  This request timed out!")

    # Example 2: Request without timeout
    print("\n" + "="*80)
    print("\nRunning generation without timeout...")

    sampling_params_no_timeout = SamplingParams(
        temperature=0.8,
        max_tokens=100,
        max_execution_time=None,  # No timeout
    )

    outputs = llm.generate(prompts[:1], sampling_params_no_timeout)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        finish_reason = output.outputs[0].finish_reason

        print(f"\nPrompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}")
        print(f"Finish reason: {finish_reason}")

    print("\n" + "="*80)
    print("\nKey points about max_execution_time:")
    print("1. Measured from request arrival time (includes queue wait)")
    print("2. Uses time.perf_counter() for monotonic, high-precision timing")
    print("3. Checked every scheduler step (~10-100ms intervals)")
    print("4. Non-blocking - no GPU performance impact")
    print("5. Works for both streaming and non-streaming requests")
    print("6. Returns finish_reason='timeout' when triggered")


if __name__ == "__main__":
    main()
