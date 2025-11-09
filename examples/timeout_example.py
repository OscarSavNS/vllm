"""Example demonstrating the request timeout feature in vLLM.

This example shows how to use max_execution_time to set graceful timeouts
on generation requests that return partial results when triggered.
"""

from vllm import LLM, SamplingParams


def main():
    llm = LLM(model="facebook/opt-125m")

    # Example 1: Generation with very short timeout to demonstrate the feature
    print("Example 1: Generation with 0.5-second timeout (will timeout)")
    print("=" * 80)

    prompts = [
        "Write a very long and detailed story about a dragon who lived in a mountain:",
    ]

    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=500,  # Request many tokens
        max_execution_time=0.5,  # Very short timeout - will trigger
    )

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(f"\nPrompt: {output.prompt!r}")
        generated_text = output.outputs[0].text
        print(f"Generated ({len(generated_text)} chars): {generated_text[:100]!r}...")
        print(f"Finish reason: {output.outputs[0].finish_reason}")
        print(f"Tokens generated: {len(output.outputs[0].token_ids)}")

        if output.outputs[0].finish_reason == "timeout":
            print("✓ Request correctly timed out and returned partial results!")
        else:
            print(f"⚠️  Expected timeout but got: {output.outputs[0].finish_reason}")

    # Example 2: Generation with generous timeout (will complete normally)
    print("\n" + "=" * 80)
    print("Example 2: Generation with 30-second timeout (will complete)")
    print("=" * 80)

    simple_prompt = ["What is the capital of France?"]

    sampling_params_generous = SamplingParams(
        temperature=0.0,
        max_tokens=20,
        max_execution_time=30.0,  # Generous timeout
    )

    outputs = llm.generate(simple_prompt, sampling_params_generous)

    for output in outputs:
        print(f"\nPrompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
        print(f"Finish reason: {output.outputs[0].finish_reason}")

        if output.outputs[0].finish_reason != "timeout":
            print("✓ Request completed normally within timeout window")
        else:
            print("⚠️  Unexpected timeout")

    print("\n" + "=" * 80)
    print("\nKey points:")
    print("- Timeout measured from request arrival (includes queue wait)")
    print("- Returns partial results with finish_reason='timeout'")
    print("- Set to None or omit for no timeout (default)")
    print("- Useful for SLO enforcement and resource management")


if __name__ == "__main__":
    main()
