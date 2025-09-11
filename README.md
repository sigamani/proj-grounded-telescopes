# A High Throughput LLM Batch Server using RayData with vLLM

[![CI Pipeline]([https://github.com/michaelsigamani/proj-grounded-telescopes/actions/workflows/ci.yml/badge.svg)](https://github.com/michaelsigamani/proj-grounded-telescopes/actions/workflows/ci.yml](https://github.com/sigamani/proj-grounded-telescopes/blob/main/.github/workflows/ci.yml))
[![Docker Hub](https://img.shields.io/docker/pulls/michaelsigamani/proj-grounded-telescopes)](https://hub.docker.com/r/michaelsigamani/proj-grounded-telescopes)

Production AI inference pipeline built on Ray + vLLM with monitoring and testing.

## Quick Start

### Development
```bash
docker compose -f compose.dev.yaml up --build
```

### Production
```bash
docker compose -f compose.prod.yaml up
```

Access:
- Ray Dashboard: http://localhost:8265
- Prometheus: http://localhost:9090 (dev only)
- Grafana: http://localhost:3000 (dev only)

<details>
<summary>Features</summary>

- Ray cluster distributed computing
- vLLM high-performance LLM inference
- NVIDIA CUDA GPU support
- Prometheus + Grafana + Loki monitoring stack
- Automated testing in CI/CD pipeline
- Pre-built Docker Hub images

</details>

<details>
<summary>Architecture</summary>

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client/API    │───▶│   Ray Head Node   │───▶│ vLLM Inference  │
│                 │    │  (Scheduler)      │    │   Workers       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌────────┴────────┐
                       │   Monitoring    │
                       │ (Prometheus/    │
                       │  Grafana/Loki)  │
                       └─────────────────┘
```

### Monitoring (Development)
- Prometheus: Metrics collection and alerting
- Grafana: Visualization and dashboards  
- Loki: Log aggregation and analysis
- Promtail: Log shipping agent

</details>

<details>
<summary>Usage</summary>

### Ray Jobs API

```python
import requests

job_data = {
    "entrypoint": "python src/batch_infer.py",
    "runtime_env": {"working_dir": "."}
}

response = requests.post("http://localhost:8265/api/jobs/", json=job_data)
job_info = response.json()
print(f"Job ID: {job_info['job_id']}")
```

</details>

<details>
<summary>Testing</summary>

```bash
# Unit tests
pytest tests/unit/ -v

# End-to-end tests  
pytest tests/e2e/ -v

# All tests with coverage
pytest --cov=src tests/
```

</details>

<details>
<summary>Configuration</summary>

### Environment Variables
- `RAY_ADDRESS`: Ray cluster address (default: "auto")
- `CUDA_VISIBLE_DEVICES`: GPU device selection
- `RAY_DISABLE_IMPORT_WARNING`: Suppress Ray warnings

### Model Configuration
Edit `src/batch_infer.py`:
```python
cfg = vLLMEngineProcessorConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",
    engine_kwargs={"max_model_len": 16384},
    concurrency=1, 
    batch_size=64,
)
```

</details>

<details>
<summary>Monitoring</summary>

### Metrics Available
- Ray cluster resources and task execution
- vLLM inference throughput and latency  
- Container resource usage
- GPU utilization

### Dashboards
- Ray Dashboard: http://localhost:8265
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

</details>

<details>
<summary>Deployment</summary>

### Two-Stage Process

**Stage 1: Build & Test**
1. CI/CD builds image from NVIDIA/PyTorch base
2. Adds Ray 2.49.1 + vLLM 0.10.0 + dependencies  
3. Runs test suite
4. Pushes to Docker Hub as version 0.1.1 on success

**Stage 2: Production**
5. Production deploys tested image from Docker Hub
6. Runs Ray cluster with job submission capability

### CI/CD Pipeline

**On Pull Request:**
- Code linting and formatting
- Unit and integration tests
- Security vulnerability scanning
- OPA policy validation
- Docker build verification

**On Main Branch Push:**
- All PR checks plus full end-to-end testing
- Multi-architecture build (arm64)
- Push to Docker Hub with version tags

</details>

<details>
<summary>Security</summary>

- Dependency vulnerability scanning (Safety + Bandit)
- OPA policy enforcement for container security
- Resource limits and isolation
- Minimal attack surface in production images

</details>


<details>
<summary>Links</summary>

- [Docker Hub Repository](https://hub.docker.com/r/michaelsigamani/proj-grounded-telescopes)
- [Ray Documentation](https://docs.ray.io/)
- [vLLM Documentation](https://docs.vllm.ai/)

</details>

<details>
<summary>Requirements</summary>

- Docker & Docker Compose (latest docker compose), vLLM 0.10.0 and Ray 2.49.1
- NVIDIA Docker Runtime (for GPU support) with nvidia-smi and CUDA drivers installed
- 24GB+ VRAM minimum (RTX 3090 or equivalent for 8B Llama models)
- CUDA 12 or higher (required, not optional)

</details>

