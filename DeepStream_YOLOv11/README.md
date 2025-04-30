![Untitled Diagram drawio (1)](https://github.com/user-attachments/assets/5d908fd1-b60d-4cd2-b318-b420bd113b3b)# Model Deployment in Jetson Orin Nano

### Supported Platforms 
- Ubuntu 20.04
- Jetson platform (Orin Nano, Xavier, etc.)
- DeepStream SDK 6.3
- Python 3.10

**Note: Other OS/DeepStream versions may require minor modifications.**

### Installation

1. Clone the repository

```bash
git clone https://github.com/Njion-v/DrinkScan.git
cd DeepStream-YOLOv11
```

2. Download required libraries

Before building the Docker image, make sure the necessary Python dependencies are downloaded:

```bash
# Download the pyds Python bindings
wget https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.8/pyds-1.1.8-py3-none-linux_x86_64.whl
```

The .whl file must be placed in the directory DeepStream-YOLOv11.

3. Pull the DeepStream Docker container

Run the following command to download the official NVIDIA DeepStream container:

```bash
docker pull nvcr.io/nvidia/deepstream:6.3-triton-multiarch
```

**Tip: If you already have the image, you can skip this step.**

4. Build the custom DrinkScan Docker image

```bash
docker build -t drinkscancan:latest .
```

5. Configure display (for GUI applications)

If you're running the application inside Docker and need to render GUI, you'll have to set up display access:

   1. On the host system, run these commands to grant the Docker container access to the X11 display:

      ```bash
      xhost +SI:localuser:root
      xhost +SI:localuser:$(whoami)
      ```

      **These commands allow the Docker container to access the host system's X11 server (needed for GUI rendering).**

   2. Start the Docker container with the following command:

      ```bash
      docker run \
          docker run  -d   -it --rm     --runtime nvidia -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/video0 --device /dev/video2 -v ./:/deepstream     -w /deepstream     --name deepstream-101     -p 8554:8554     deepstream-yolo:latest
      ```

   3. Inside the running Docker container, set the display environment:

      Check the display with xrandr:

      ```bash
      xrandr --query
      ```

      If this shows no display, you'll need to set the DISPLAY variable manually.

      Set the DISPLAY environment variable to match the display of the host system:

      ```bash
      export DISPLAY=:1
      ```

      **Note: If you're using a different display (e.g., virtual display), adjust :0 to :1 or whatever corresponds to your setup.**

## Overview 

The deepstream_usb_camera.py follows this pipeline:
![Untitled Diagram drawio (1)](https://github.com/user-attachments/assets/1e7fb801-489f-4c70-8735-b1f6ea8f16fc)

  - v4l2src: Captures video from a camera or other video sources
  - capsfilter: Defines the format and properties of the incoming video stream (resolution, pixel format).
  - videoconvert: Converts video pixel formats on the CPU (YUYV to I420).
  - nvvideoconvert: Converts video pixel formats and memory on the GPU for better performance (I420 to NV12, host-memory to NVMM).
  - capsfilter: Ensures the video format (NV12) is compatible with the next stages, using GPU memory (NVMM).
  - nvstreammux: Batches frames for efficient processing, resizes and pads the input frames to match the model's input size.
  - nvinfer: Runs a deep learning model on the batched frames
  - nvvideoconvert: Converts the post-inference video from NV12 (GPU memory format) to RGB (for display purposes).
  - nvdosd: Overlays information (bounding boxes, labels) on the video.
  - nv3dsink: Render to external monitor.

## Running the application
   ```bash
   cd apps/deepstream-test1
   # Run the application with usb camera
   python3 deepstream_3_cam.py
   ```
