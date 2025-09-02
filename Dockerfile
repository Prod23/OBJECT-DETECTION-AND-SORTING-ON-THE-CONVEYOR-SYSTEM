# Multi-stage Docker build for conveyor object detection system
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libgl1-mesa-glx \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    ffmpeg \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM base as production

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed runs logs notebooks

# Set permissions
RUN chmod +x scripts/*.py

# Expose ports for web services (if needed)
EXPOSE 8000

# Default command
CMD ["python", "scripts/infer.py", "--help"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir jupyter notebook ipykernel

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed runs logs notebooks

# Set permissions
RUN chmod +x scripts/*.py

# Install kernel for Jupyter
RUN python -m ipykernel install --user --name=conveyor-detection

# Expose Jupyter port
EXPOSE 8888 8000

# Default command for development
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
