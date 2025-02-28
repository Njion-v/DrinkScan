import cv2
import numpy as np
import os
import threading
import time
from datetime import datetime
from ultralytics import YOLO

class MultiCameraYOLO:
    def __init__(self, camera_ids=[0, 1, 3]):
        self.camera_ids = camera_ids
        self.cameras = {}
        self.frames = {}
        self.running = True
        self.frame_width = 640  # Độ phân giải tối ưu
        self.frame_height = 480
        self.output_dir = "captured_images_yolo"
        os.makedirs(self.output_dir, exist_ok=True)

        # Load mô hình YOLO
        self.model = YOLO("models/Yolov11-24.2/detect/train/weights/best.pt")

        self._init_cameras()
        self.capture_threads = []
        self._start_capture_threads()

    def _init_cameras(self):
        for cam_id in self.camera_ids:
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                self.cameras[cam_id] = cap
                self.frames[cam_id] = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            else:
                print(f"Cảnh báo: Không thể mở camera {cam_id}")

    def _capture_thread(self, camera_id):
        while self.running:
            if camera_id in self.cameras:
                ret, frame = self.cameras[camera_id].read()
                if ret:
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                    frame = self._process_frame(frame)
                    self.frames[camera_id] = frame
                time.sleep(0.033)  # ~30 FPS

    def _process_frame(self, frame):
        """Nhận diện đối tượng với YOLO, chỉ hiển thị box nếu confidence > 0.8"""
        results = self.model(frame)
        for result in results:
            for box in result.boxes:
                confidence = box.conf[0].item()
                if confidence > 0.7:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = result.names[int(box.cls[0])]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def _start_capture_threads(self):
        for cam_id in self.cameras.keys():
            thread = threading.Thread(target=self._capture_thread, args=(cam_id,))
            thread.daemon = True
            thread.start()
            self.capture_threads.append(thread)

    def capture_images(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for cam_id, frame in self.frames.items():
            if frame is not None:
                filename = f"{self.output_dir}/yolov11s_camera_{cam_id}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
        print(f"Ảnh được chụp vào {timestamp}")

    def run(self):
        """Hiển thị tất cả khung hình camera."""
        while self.running:
            frames = [self.frames.get(cam_id, np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8))
                      for cam_id in self.camera_ids]

            # Sắp xếp lại layout: 3 camera xếp theo hàng ngang
            combined_frame = np.hstack(frames) if len(frames) == 3 else None

            if combined_frame is not None:
                cv2.imshow("Multi-Camera YOLO Detection", combined_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord(' '):
                self.capture_images()
        self._cleanup()

    def _cleanup(self):
        self.running = False
        for cap in self.cameras.values():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_system = MultiCameraYOLO()
    capture_system.run()
