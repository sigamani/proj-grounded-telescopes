Intended dir structure


├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── /src               # optional: put batch_infer.py / preprocessing (not written but will be pydantic serialisation of pii) / utilities code and stuff
├── /monitoring        # configs for Grafana / Prometheus / Loki
│    ├── prometheus.yml
│    ├── dashboards/
│    └── ...
├── /ray-tmp           # host-volume for Ray’s spill/session directories (dunno if this is a great place due to size constraints maybe /tmp?)
└── .dockerignore      (yet to be birthed)

This repository contains the code, infrastructure, and configuration to build, deploy, and monitor an AI inference / LLM pipeline, based on Ray + vLLM. Core features include:

Component Purpose

Ray Head Service Master node for Ray cluster. Handles task scheduling, dashboards, and cluster coordination. Haiku (Inference) Service Client/worker service that processes prompts via vLLM, generates haiku output. Dockerfile(s) Defines container image builds: base image, Python dependencies (Ray, vLLM, etc.), and entrypoints. Docker Compose Setup Orchestrates multi-container stack including Ray head + inference service (haiku). GPU Support Uses NVIDIA base images, requires container toolkit, configured to detect and use CUDA. Monitoring Stack Prometheus + Grafana + Loki/Promtail to collect metrics and logs of Ray + inference processes. Configuration Files Environment variables, Ray init settings, spill directories, and port mappings to manage performance and resource usage.

Usage

Build the Docker image locally or push to Docker Hub.
Start services via docker compose up --build.
Access Ray dashboard (http://localhost:8265⁠), model API endpoints (e.g. http://localhost:8000⁠), etc.
Enable monitoring/logs via added stack to track performance, logs, and system metrics.
                                  +---------------------+
                                  | Client / API Caller |
                                  +----------+----------+
                                             |
                                       HTTP / RPC
                                             |
                                  +----------v----------+
                                  |     Ray Head Node    |
                                  | (Scheduler, Metrics, |
                                  |   Dashboard, etc.)   |
                                  +----------+-----------+
                                             |
            +--------------------------------+---------------------------------+
            |                                |                                 |
    +-------v-------+              +---------v---------+        +--------------v--------------+
    | Worker: vLLM   |              | Worker: Tokenizer /  |        | Object Store / Spill / KV |
    |  Inference     |              | Preprocessor        |        | Cache / GPU Memory          |
    +-------+--------+              +---------------------+        +-----------------------------+
            |                                |                                 |
            +--------------------------------+---------------------------------+
                                             |
                        +--------------------+--------------------+
                        | Monitoring & Logging                   |
                        |                                          |
                +-------v------+        +------------------v------------------+
                | Metrics      |        |   Logs                            |
                | Prometheus   |        | Loki / Promtail                   |
                +--------------+        +-----------------------------------+
                        |
                +-------v-------+
                | Grafana       |
                | Dashboards    |
                +---------------+
Key Dependencies & Versions

Python 3.10
Ray = 2.49.1 with ray[data] for LLM pipelines
vLLM compatible version = 0.10.0 - updating to 0.10.0 will trigger a got an unexpected keyword argument 'disable_log_requests' known bug
CUDA-capable environment via proper base image (e.g. NVIDIA PyTorch image) + container runtime configured for GPUs
Known Limitations

Spill directory and object store memory can fill up quickly. Requires sufficient disk and memory.
Version mismatches between Ray & vLLM flags (such as disable_log_requests) have caused errors — versions must match exactly.
Permissions issues with mounted volume dirs (e.g. /tmp/ray) may require setup of host directories with correct ownership.
