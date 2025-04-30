# Use NVIDIA DeepStream base image
FROM nvcr.io/nvidia/deepstream:6.3-triton-multiarch

# Set working directory
WORKDIR /deepstream

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install required packages
RUN apt update && apt install -y \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    libavcodec58 \
    libmpg123-0 \
    libmpeg2encpp-2.1-0 \
    libmpeg2-4 \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    v4l-utils \
    x11-xserver-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY pyds-1.1.8-py3-none-linux_aarch64.whl /deepstream/
RUN pip3 install /deepstream/pyds-1.1.8-py3-none-linux_aarch64.whl && \
    rm /deepstream/pyds-1.1.8-py3-none-linux_aarch64.whl
RUN pip3 install cuda-python

# Set display environment variable
ENV DISPLAY=:1

# Expose necessary ports
EXPOSE 8554

# Set entrypoint
ENTRYPOINT ["/bin/bash"]
