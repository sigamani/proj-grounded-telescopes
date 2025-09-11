FROM nvcr.io/nvidia/pytorch:24.06-py3 

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files to the container
COPY . /app/

# Set environment variables
ENV RAY_ADDRESS=auto

# Expose Ray ports
EXPOSE 6379 8265 10001 8000 8888

# Default command
CMD ["ray", "start", "--head", "--dashboard-host=0.0.0.0", "--port=6379", "--dashboard-port=8265", "--block"] 
