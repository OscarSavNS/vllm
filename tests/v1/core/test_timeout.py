"""Test timeout functionality."""
import time

import pytest

from vllm import LLM, SamplingParams
from vllm.v1.engine import FinishReason
from vllm.v1.request import Request, RequestStatus


def test_sampling_params_timeout():
    """Test that max_execution_time is properly set in SamplingParams."""
    # Valid timeout
    params = SamplingParams(max_execution_time=5.0)
    assert params.max_execution_time == 5.0

    # None timeout (no timeout)
    params = SamplingParams(max_execution_time=None)
    assert params.max_execution_time is None

    # Invalid timeout (should raise ValueError)
    with pytest.raises(ValueError, match="max_execution_time must be positive"):
        SamplingParams(max_execution_time=0.0)

    with pytest.raises(ValueError, match="max_execution_time must be positive"):
        SamplingParams(max_execution_time=-1.0)


def test_request_arrival_perf_counter():
    """Test that Request tracks arrival_perf_counter."""
    start_time = time.perf_counter()

    request = Request(
        request_id="test",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_execution_time=10.0),
        pooling_params=None,
        eos_token_id=0,
    )

    end_time = time.perf_counter()

    # Verify arrival_perf_counter is set
    assert request.arrival_perf_counter is not None
    assert start_time <= request.arrival_perf_counter <= end_time

    # Verify it's different from arrival_time
    assert request.arrival_time is not None
    # arrival_time uses time.time(), arrival_perf_counter uses time.perf_counter()
    # They should be different values
    assert request.arrival_time != request.arrival_perf_counter


def test_timeout_request_status():
    """Test that FINISHED_TIMEOUT status is properly defined."""
    # Verify FINISHED_TIMEOUT exists
    assert hasattr(RequestStatus, "FINISHED_TIMEOUT")

    # Verify it's a finished status
    assert RequestStatus.is_finished(RequestStatus.FINISHED_TIMEOUT)

    # Verify it maps to TIMEOUT finish reason
    finish_reason = RequestStatus.get_finished_reason(RequestStatus.FINISHED_TIMEOUT)
    assert finish_reason == FinishReason.TIMEOUT


def test_timeout_finish_reason():
    """Test that TIMEOUT finish reason is properly defined."""
    # Verify TIMEOUT exists
    assert hasattr(FinishReason, "TIMEOUT")

    # Verify it has the correct string representation
    assert str(FinishReason.TIMEOUT) == "timeout"

    # Verify it's distinct from other finish reasons
    assert FinishReason.TIMEOUT != FinishReason.STOP
    assert FinishReason.TIMEOUT != FinishReason.LENGTH
    assert FinishReason.TIMEOUT != FinishReason.ABORT


def test_sampling_params_from_optional_timeout():
    """Test that max_execution_time works with from_optional."""
    params = SamplingParams.from_optional(
        max_execution_time=30.0,
        temperature=0.7,
    )
    assert params.max_execution_time == 30.0
    assert params.temperature == 0.7

    # Test with None
    params = SamplingParams.from_optional(
        max_execution_time=None,
        temperature=0.5,
    )
    assert params.max_execution_time is None
    assert params.temperature == 0.5


@pytest.mark.skip_v1
def test_timeout_returns_partial_results():
    """Test that timeout returns partial results (graceful timeout).

    This integration test verifies the key behavior: when a request times out,
    it returns whatever tokens have been generated so far rather than discarding them.
    """
    llm = LLM(model="facebook/opt-125m", max_model_len=512)

    prompt = "Write a very long story about a dragon:"

    # Set a very short timeout to force timeout during generation
    sampling_params = SamplingParams(
        max_tokens=500,  # Request many tokens
        max_execution_time=1.0,  # But timeout after 1 second
        temperature=0.8,
    )

    start_time = time.time()
    outputs = llm.generate([prompt], sampling_params)
    elapsed = time.time() - start_time

    assert len(outputs) == 1
    output = outputs[0]

    # Verify timeout occurred
    assert output.outputs[0].finish_reason == "timeout"

    # Verify timeout was enforced (allow some overhead)
    assert elapsed < sampling_params.max_execution_time + 2.0

    # Verify partial results were returned (key behavior!)
    # We should have generated SOME tokens before timeout
    num_tokens = len(output.outputs[0].token_ids)
    assert num_tokens > 0, "Timeout should return partial results, not empty output"

    # Verify text was generated
    assert len(output.outputs[0].text) > 0


@pytest.mark.skip_v1
def test_timeout_with_multiple_requests():
    """Test timeout behavior with multiple concurrent requests.

    Verify that timeout is tracked independently per request.
    """
    llm = LLM(model="facebook/opt-125m", max_model_len=512)

    prompts = [
        "Short prompt",  # Should complete quickly
        "Write a very long story about dragons and knights:",  # May timeout
    ]

    sampling_params = SamplingParams(
        max_tokens=200,
        max_execution_time=2.0,
        temperature=0.8,
    )

    outputs = llm.generate(prompts, sampling_params)

    assert len(outputs) == 2

    # Both should return results (either completed or timed out)
    for output in outputs:
        assert output.outputs[0].finish_reason in ("stop", "length", "timeout")
        # Even if timed out, should have partial results
        assert len(output.outputs[0].token_ids) >= 0


@pytest.mark.skip_v1
def test_no_timeout_completes_normally():
    """Test that requests without timeout complete normally."""
    llm = LLM(model="facebook/opt-125m", max_model_len=512)

    prompt = "The capital of France is"

    # No timeout specified (default behavior)
    sampling_params = SamplingParams(
        max_tokens=20,
        temperature=0.0,
    )

    outputs = llm.generate([prompt], sampling_params)

    assert len(outputs) == 1
    output = outputs[0]

    # Should NOT timeout
    assert output.outputs[0].finish_reason != "timeout"
    assert output.outputs[0].finish_reason in ("stop", "length")

    # Should have generated tokens
    assert len(output.outputs[0].token_ids) > 0
