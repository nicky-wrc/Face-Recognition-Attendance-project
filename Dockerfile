# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    python3-opencv \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    liblapack-dev \
    libopenblas-dev \
    gfortran \
    wget \
    curl \
    bzip2 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set timezone (system-level)
ENV TZ=Asia/Bangkok
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Download dlib models if they don't exist
RUN if [ ! -f "shape_predictor_68_face_landmarks.dat" ]; then \
        wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
        bunzip2 shape_predictor_68_face_landmarks.dat.bz2; \
    fi

RUN if [ ! -f "dlib_face_recognition_resnet_model_v1.dat" ]; then \
        wget http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2 && \
        bunzip2 dlib_face_recognition_resnet_model_v1.dat.bz2; \
    fi

# Create necessary directories
RUN mkdir -p dataset uploads static/images

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Run the application
CMD ["python", "app.py"]