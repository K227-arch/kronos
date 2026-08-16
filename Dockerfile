# Kronos Web UI - Docker Image
# Uses NVIDIA CUDA base for GPU inference support

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python3 as default
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt /app/requirements.txt
COPY webui/requirements.txt /app/webui_requirements.txt

# Install Python dependencies
# Install PyTorch with CUDA support, then the rest
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
    flask \
    flask-cors \
    pandas \
    numpy \
    plotly \
    huggingface_hub \
    einops \
    safetensors \
    tqdm \
    matplotlib

# Copy application code
COPY model/ /app/model/
COPY webui/app.py /app/webui/app.py
COPY webui/templates/ /app/webui/templates/

# Create directories for runtime data
RUN mkdir -p /app/webui/prediction_results /app/data

# Set HuggingFace cache directory (mountable as volume)
ENV HF_HOME=/app/.cache/huggingface

# Expose the web UI port
EXPOSE 7070

# Run from project root so model imports work
WORKDIR /app
CMD ["python", "webui/app.py"]
