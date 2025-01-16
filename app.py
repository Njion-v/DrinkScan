# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import os
from datetime import datetime
import json

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/images'
ANNOTATION_FOLDER = 'static/annotations'
CAMERA_COUNT = 3

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATION_FOLDER, exist_ok=True)

# Initialize cameras
cameras = []
for i in range(CAMERA_COUNT):
    try:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(cap)
    except Exception as e:
        print(f"Failed to open camera {i}: {e}")

# Categories and labels
CATEGORIES = {
    'type': ['Bottle', 'Can'],
    'brand': ['Coca', 'Water', 'Beer']
}

@app.route('/')
def index():
    return render_template('index.html', categories=CATEGORIES)

@app.route('/capture/<int:camera_id>', methods=['POST'])
def capture_image(camera_id):
    if camera_id < len(cameras):
        cap = cameras[camera_id]
        ret, frame = cap.read()
        if ret:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_{camera_id}_{timestamp}.jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            cv2.imwrite(filepath, frame)
            return jsonify({'success': True, 'filename': filename})
    return jsonify({'success': False, 'error': 'Failed to capture image'})

@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    data = request.json
    filename = f"{data['image'].split('.')[0]}_annotation.json"
    filepath = os.path.join(ANNOTATION_FOLDER, filename)
    
    with open(filepath, 'w') as f:
        json.dump(data, f)
    
    return jsonify({'success': True})

@app.route('/images')
def get_images():
    images = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(('.jpg', '.png'))]
    return jsonify(images)

if __name__ == '__main__':
    app.run(debug=True)