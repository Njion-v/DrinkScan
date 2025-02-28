from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from datetime import datetime
import os
import threading
import time

app = Flask(__name__)

class MultiCameraCapture:
    def __init__(self, camera_ids: list = [0, 3, 2]):
        self.camera_ids = camera_ids
        self.cameras: dict = {}
        self.frames: dict = {}
        self.running = True
        self.frame_width = 1920
        self.frame_height = 1080
        self.output_dir = "captured_images_new"
        os.makedirs(self.output_dir, exist_ok=True)
        self._init_cameras()
        self._start_capture_threads()

    def _init_cameras(self):
        for cam_id in self.camera_ids:
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                print(f"Warning: Unable to open camera {cam_id}")
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cameras[cam_id] = cap
            self.frames[cam_id] = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

    def _capture_thread(self, camera_id: int):
        while self.running:
            ret, frame = self.cameras[camera_id].read()
            if ret:
                self.frames[camera_id] = cv2.resize(frame, (self.frame_width, self.frame_height))
            time.sleep(0.033)

    def _start_capture_threads(self):
        for cam_id in self.cameras.keys():
            thread = threading.Thread(target=self._capture_thread, args=(cam_id,))
            thread.daemon = True
            thread.start()

    def capture_images(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for cam_id, frame in self.frames.items():
            if frame is not None:
                filename = f"{self.output_dir}/camera_{cam_id}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
        print(f"Images captured at {timestamp}")

    def get_frame(self, camera_id: int):
        return self.frames.get(camera_id, None)

    def cleanup(self):
        self.running = False
        for cap in self.cameras.values():
            cap.release()

# Initialize the camera system
camera_system = MultiCameraCapture()

@app.route('/')
def index():
    return render_template('index.html', camera_ids=camera_system.camera_ids)

def generate_frames(camera_id):
    while True:
        frame = camera_system.get_frame(camera_id)
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    camera_system.capture_images()
    return jsonify(success=True)

@app.route('/cleanup', methods=['POST'])
def cleanup():
    camera_system.cleanup()
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)