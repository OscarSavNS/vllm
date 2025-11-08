"""Debug script for timeout feature using sync engine with ipdb breakpoints."""

from vllm import LLM, SamplingParams
import ipdb


def main():
    # Initialize the LLM
    print("Initializing LLM...")
    llm = LLM(
        model="facebook/opt-125m",
        enforce_eager=True,
        gpu_memory_utilization=0.5,
    )

    # Simple test case with very short timeout to trigger the issue
    prompts = ["Write a very long story about a dragon that lived in a mountain."]

    # Set a very short timeout to ensure it triggers
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=200,
        max_execution_time=2,  # Very short timeout - 2 seconds
    )

    print("Running generation with 0.5-second timeout...")
    print("This should timeout and return partial results with finish_reason='timeout'")

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        finish_reason = output.outputs[0].finish_reason

        print(f"\nPrompt: {prompt!r}")
        print(f"Generated text length: {len(generated_text)} chars")
        print(f"Generated text: {generated_text!r}")
        print(f"Finish reason: {finish_reason}")

        if finish_reason == "timeout":
            print("✓ Request correctly timed out!")
        else:
            print(f"✗ Expected timeout but got: {finish_reason}")


if __name__ == "__main__":
    main()
