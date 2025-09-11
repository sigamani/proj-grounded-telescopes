FROM nvcr.io/nvidia/pytorch:24.06-py3 

# Set working directory
WORKDIR /app


# Install python packages 
RUN pip install --no-cache-dir \
    "ray[data]==2.49.1" "pyarrow>=12,<16" "pydantic>=2.6,<3" "vllm==0.10.0" \
    pandas numpy transformers openai tqdm huggingface_hub datasets \
    sentencepiece accelerate "tokenizers>=0.13.3"

# Copy project files to the container
COPY . /app/

# Set environment variables
ENV RAY_ADDRESS=auto

# Expose Ray ports
EXPOSE 6379 8265 10001 8000 8888

# Default command
CMD ["ray", "start", "--head", "--dashboard-host=0.0.0.0", "--port=6379", "--dashboard-port=8265", "--block"] 
