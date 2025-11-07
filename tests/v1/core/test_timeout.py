"""Test timeout functionality."""
import time

import pytest

from vllm.sampling_params import SamplingParams
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
