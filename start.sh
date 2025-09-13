#!/bin/bash

set -e

echo "Starting Ray Head..."
export RAY_TMPDIR=/tmp/ray_tmp
mkdir -p $RAY_TMPDIR
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir=$RAY_TMPDIR

echo "Waiting for Ray to be ready..."
sleep 5

echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model ${VLLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct} \
    --served-model-name ${VLLM_SERVED_NAME:-qwen-1.5b} \
    --port ${VLLM_PORT:-8000} \
    --dtype ${VLLM_DTYPE:-auto} \
    --trust-remote-code \
    --host 0.0.0.0