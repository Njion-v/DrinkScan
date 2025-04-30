# YOLOv11-detection ONNX TensorRT Exporter

### Convert model

#### 1. Install the requirements
```
cd models/yolov11-det
pip3 install -r requirements.txt
```

#### 2. Export ONNX model
```bash
python3 export_yolov11_det.py -w bread.pt
```

#### 3. Export TensorRT model
```bash
trtexec --onnx=./bread.onnx \
        --saveEngine=./yolo11n_b1_fp16.engine \
        --fp16
```

#### 4. Copy the generated files
Copy the generated ONNX model file, TensorRT model file, and `labels.txt` to the `configs` folder.
