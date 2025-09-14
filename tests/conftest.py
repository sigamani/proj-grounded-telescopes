"""
Pytest configuration and shared fixtures for Ray + vLLM testing.
"""

import tempfile
from unittest.mock import Mock, patch

import pytest
import ray


@pytest.fixture(scope="session")
def ray_cluster():
    """Initialize a local Ray cluster for testing."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=1, object_store_memory=100_000_000)
    try:
        yield None
    finally:
        if ray.is_initialized():
            try:
                ray.shutdown()
            except ImportError:
                # Handle Ray shutdown circular import issue in tests
                pass


@pytest.fixture
def temp_work_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_vllm_processor():
    """Mock vLLM processor for unit tests without GPU dependencies."""
    with patch("ray.data.llm.build_llm_processor") as mock_build:
        mock_processor = Mock()
        mock_processor.return_value = Mock()
        mock_build.return_value = mock_processor
        yield mock_processor


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing."""
    return [
        {"prompt": "Write a short summary of machine learning."},
        {"prompt": "Explain quantum computing in simple terms."},
        {"prompt": "What is the future of AI?"},
    ]


@pytest.fixture
def expected_response_structure():
    """Expected structure of vLLM responses."""
    return {"out": str, "prompt": str, "generated_text": str}


@pytest.fixture(scope="session")
def docker_compose_env():
    """Environment variables for docker compose testing."""
    return {
        "RAY_ADDRESS": "ray://localhost:10001",
        "CUDA_VISIBLE_DEVICES": "0",  # Mock GPU for CI
    }
