# DrinkScan - YOLOv11 applied for pastry recognition

### Set up the dependencies

- All the required dependencies are listed in `requirements.txt` file, run the following command to install the dependencies using pip.

```sh
pip install -r requirements.txt
```

### Inference

- By default, the inference uses three cameras at index 0,1,2 as the input stream.You can configure this camera index by the variable `camera_ids` in the `main.py` script.

- Run the main script via

```sh
python main.py
```

- The camera stream and YOLOv11 model takes around 3 minutes for initialization, after that, there will be a window represents the frame from the camera capture including the detected bounding boxes for beverages items.

### Evaluation

- We use Ultralytics built-in YOLO DetectionValidator for the model evaluation on a test dataset.

#### Prepare your test dataset

- Your datasets should be prepared inside the `/datasets` directory, in YOLO format.

- There should be 3 datasets: train, validation and test. The dataset should contain 2 sub-directories: `/images` for the images and the annotations in `/labels`.

- In the end, your train dataset should be at `/datasets/train`, validation set should be at `/datasets/valid` while your test dataset should be at `/datasets/test`.

#### Run the evaluation

- The evaluation script is prepared at `evaluation.py`, run it via

```sh
python evaluation.py
```

- The script will do the evaluation on your test dataset at `/datasets/test` and output the results including Precision, Recall, mAP50, mAP50-95 and some visualizations.

- The results will be in a sub directory inside `runs/detect` directory, each run will produce a separate directory storing the results.

- The evaluation process produces some visualizations such as: 
  - confusion matrix.
  - curves for: F1, Precision, Recall, and Precision-Recall.
  - example instances from the test dataset, including the true labels and the predicted boxes.

### Model Deployment in Jetson Orin Nano

- The DrinkScan YOLO model can be integrated to Jetson Orin Nano device. The details is at [DeepStream Deployment](./DeepStream-YOLOv11/README.md)

### Model Deployment as Flask API

- The DrinkScan YOLO model can be run as an API via Flask framework. The details is at [Flask API Deployment](./flask-app/README.md)


