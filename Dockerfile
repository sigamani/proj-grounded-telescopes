FROM rayproject/ray:latest-py311

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install python packages from requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir \
      "ray[data]>=2.9.0" \
      "vllm>=0.6.0,<0.7.0" \
      "pandas>=2.0.0" \
      "numpy>=1.23.0" \
      "transformers>=4.30.0" \
      "openai>=1.0.0" \
      "tqdm>=4.65.0" \
      && rm -rf ~/.cache/pip

# Copy project files to the container
COPY . /app/

# Set environment variables
ENV RAY_ADDRESS=auto
ENV OPENCV_IO_ENABLE_JASPER=0
ENV VLLM_DISABLE_CUSTOM_ALL_REDUCE=1

# Expose Ray ports
EXPOSE 6379 8265 10001 8000 8888

# Default command
CMD ["ray", "start", "--head", "--dashboard-host=0.0.0.0", "--port=6379", "--dashboard-port=8265", "--block"] 
