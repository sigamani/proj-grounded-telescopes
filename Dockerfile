FROM rayproject/ray:nightly-py39-gpu

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install python packages from requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip

# Copy project files to the container
COPY . /app/

# Set environment variables

COPY --chmod=755 start.sh /app/start.sh

# Useful env
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
ENV RAY_ADDRESS=auto
ENV RAY_ADDRESS=auto
ENV OPENCV_IO_ENABLE_JASPER=0
ENV VLLM_DISABLE_CUSTOM_ALL_REDUCE=1


# Expose Ray ports
EXPOSE 8000 8265 8080 10001 6379

CMD ["/app/start.sh"]

