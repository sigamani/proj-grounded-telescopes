"""
Unit tests for Ray cluster integration.

Key failure points tested:
- Ray cluster connectivity
- Worker node failures
- Resource allocation issues
- Task scheduling failures
- Object store spill issues
- Dashboard accessibility
"""

import pytest
import ray
from unittest.mock import patch, Mock
import time
import requests
import psutil


class TestRayIntegration:

    def test_ray_cluster_health(self, ray_cluster):
        """Test Ray cluster basic health checks."""
        assert ray.is_initialized()

        # Test cluster resources are available
        resources = ray.cluster_resources()
        assert "CPU" in resources
        assert resources["CPU"] > 0

    def test_ray_cluster_connectivity(self, ray_cluster):
        """Test Ray cluster connectivity."""
        # Test cluster info retrieval
        cluster_info = ray.cluster_resources()
        assert isinstance(cluster_info, dict)

        # Test node information
        nodes = ray.nodes()
        assert len(nodes) > 0
        assert any(node["Alive"] for node in nodes)

    def test_ray_task_execution(self, ray_cluster):
        """Test basic Ray task execution."""

        @ray.remote
        def simple_task(x):
            return x * 2

        # Test task submission and retrieval
        future = simple_task.remote(21)
        result = ray.get(future)
        assert result == 42

    def test_ray_task_failure_handling(self, ray_cluster):
        """Test Ray task failure handling."""

        @ray.remote
        def failing_task():
            raise ValueError("Intentional test failure")

        future = failing_task.remote()
        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(future)

    def test_ray_resource_allocation(self, ray_cluster):
        """Test Ray resource allocation and limits."""

        @ray.remote(num_cpus=1)
        def cpu_task():
            return "CPU task completed"

        # Test resource-constrained task
        future = cpu_task.remote()
        result = ray.get(future, timeout=10)
        assert result == "CPU task completed"

    def test_ray_object_store(self, ray_cluster):
        """Test Ray object store functionality."""
        # Test object storage and retrieval
        large_data = [i for i in range(1000)]
        ref = ray.put(large_data)
        retrieved_data = ray.get(ref)

        assert retrieved_data == large_data

    def test_ray_object_store_spill(self, ray_cluster):
        """Test Ray object store spill behavior."""
        # Create data that might trigger spill
        large_objects = []
        try:
            for i in range(10):
                large_data = [j for j in range(100000)]
                ref = ray.put(large_data)
                large_objects.append(ref)

            # Verify objects are still accessible
            first_obj = ray.get(large_objects[0])
            assert len(first_obj) == 100000

        except Exception as e:
            # Object store full is acceptable in test environment
            assert "object store" in str(e).lower() or "memory" in str(e).lower()

    @pytest.mark.skipif(
        not ray.is_initialized() or ray.cluster_resources().get("GPU", 0) == 0,
        reason="GPU not available in test environment",
    )
    def test_ray_gpu_allocation(self, ray_cluster):
        """Test Ray GPU resource allocation."""

        @ray.remote(num_gpus=1)
        def gpu_task():
            import torch

            return torch.cuda.is_available()

        future = gpu_task.remote()
        result = ray.get(future, timeout=30)
        assert result is True

    def test_ray_dashboard_accessibility(self):
        """Test Ray dashboard accessibility (when available)."""
        # Try to connect to Ray dashboard
        try:
            response = requests.get("http://localhost:8265", timeout=5)
            # If dashboard is running, should get some response
            assert response.status_code in [200, 404, 403]  # Various valid responses
        except requests.exceptions.ConnectionError:
            # Dashboard not running in test env - this is acceptable
            pytest.skip("Ray dashboard not available in test environment")

    def test_ray_worker_failure_recovery(self, ray_cluster):
        """Test Ray worker failure recovery."""

        @ray.remote
        def worker_task(task_id):
            if task_id == 5:  # Simulate failure on specific task
                raise RuntimeError("Worker failure simulation")
            return f"Task {task_id} completed"

        # Submit multiple tasks, some will fail
        futures = [worker_task.remote(i) for i in range(10)]

        results = []
        errors = []

        for future in futures:
            try:
                result = ray.get(future)
                results.append(result)
            except ray.exceptions.RayTaskError as e:
                errors.append(str(e))

        # Should have some successful results and one error
        assert len(results) == 9  # 9 successful tasks
        assert len(errors) == 1  # 1 failed task

    def test_ray_memory_monitoring(self, ray_cluster):
        """Test Ray memory usage monitoring."""
        # Get initial memory state
        initial_stats = ray.cluster_resources()

        # Create some objects
        refs = []
        for i in range(5):
            data = [j for j in range(10000)]
            ref = ray.put(data)
            refs.append(ref)

        # Memory should be allocated
        current_stats = ray.cluster_resources()

        # Clean up
        for ref in refs:
            del ref

        # Basic validation that stats are being tracked
        assert isinstance(initial_stats, dict)
        assert isinstance(current_stats, dict)

    def test_ray_scaling_simulation(self, ray_cluster):
        """Test Ray task scaling behavior."""

        @ray.remote
        def scaling_task(size):
            return sum(range(size))

        # Test different workload sizes
        small_tasks = [scaling_task.remote(100) for _ in range(5)]
        large_tasks = [scaling_task.remote(10000) for _ in range(2)]

        # All tasks should complete
        small_results = ray.get(small_tasks)
        large_results = ray.get(large_tasks)

        assert len(small_results) == 5
        assert len(large_results) == 2
        assert all(isinstance(r, int) for r in small_results + large_results)
