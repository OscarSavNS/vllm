"""Example demonstrating the request timeout feature in vLLM.

This example shows how to use max_execution_time to set graceful timeouts
on generation requests that return partial results when triggered.
"""

from vllm import LLM, SamplingParams


def main():
    llm = LLM(model="facebook/opt-125m")

    # Example 1: Generation with timeout
    print("Example 1: Generation with 5-second timeout")
    print("=" * 80)

    prompts = [
        "What is the capital of France?",
        "Write a very long story about a dragon.",
    ]

    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=1000,
        max_execution_time=5.0,  # Timeout after 5 seconds
    )

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(f"\nPrompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
        print(f"Finish reason: {output.outputs[0].finish_reason}")

        if output.outputs[0].finish_reason == "timeout":
            print("⚠️  Request timed out (partial results returned)")

    # Example 2: Generation without timeout (default behavior)
    print("\n" + "=" * 80)
    print("Example 2: Generation without timeout (default)")
    print("=" * 80)

    sampling_params_no_timeout = SamplingParams(
        temperature=0.8,
        max_tokens=50,
    )

    outputs = llm.generate(prompts[:1], sampling_params_no_timeout)

    for output in outputs:
        print(f"\nPrompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
        print(f"Finish reason: {output.outputs[0].finish_reason}")

    print("\n" + "=" * 80)
    print("\nKey points:")
    print("- Timeout measured from request arrival (includes queue wait)")
    print("- Returns partial results with finish_reason='timeout'")
    print("- Set to None or omit for no timeout (default)")
    print("- Works with both streaming and non-streaming requests")


if __name__ == "__main__":
    main()
