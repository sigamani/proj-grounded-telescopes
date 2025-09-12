FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ray and core dependencies
RUN pip install --no-cache-dir \
    "ray[data]==2.49.1" \
    "vllm==0.10.0" \
    "torch>=2.0.0" \
    "pydantic>=2.6,<3" \
    "pyarrow>=12,<16" \
    "huggingface_hub>=0.33.0" \
    "datasets==2.21.0" \
    "openai>=1.42.0" \
    "tqdm>=4.66.5"

# Create app directory
WORKDIR /app

# Copy application files
COPY src/ /app/src/
COPY requirements.txt /app/

# Expose ports
EXPOSE 6379
EXPOSE 8265
EXPOSE 10001
EXPOSE 8000
EXPOSE 8888

# Set environment variables
ENV PYTHONPATH=/app
ENV RAY_DISABLE_IMPORT_WARNING=1
ENV RAY_ADDRESS=auto

# Default command
CMD ["ray", "start", "--head", "--dashboard-host=0.0.0.0", "--port=6379", "--dashboard-port=8265", "--block"] 