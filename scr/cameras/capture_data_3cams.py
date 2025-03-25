import cv2
import numpy as np
import os
import threading
import time
from datetime import datetime

class MultiCameraYOLO:
    def __init__(self, camera_ids=[0]):
        self.camera_ids = camera_ids
        self.cameras = {}
        self.frames = {}
        self.running = True
        self.capture_width = 1920  # Capture resolution
        self.capture_height = 1080
        self.display_width = 640  # Display resolution for each square
        self.display_height = 480
        self.output_dir = "captured_images"
        os.makedirs(self.output_dir, exist_ok=True)

        # Load mô hình YOLO
        self._init_cameras()
        self.capture_threads = []
        self._start_capture_threads()

    def _init_cameras(self):
        for cam_id in self.camera_ids:
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
                self.cameras[cam_id] = cap
                self.frames[cam_id] = np.zeros((self.capture_height, self.capture_width, 3), dtype=np.uint8)
            else:
                print(f"Cảnh báo: Không thể mở camera {cam_id}")

    def _capture_thread(self, camera_id):
        while self.running:
            if camera_id in self.cameras:
                ret, frame = self.cameras[camera_id].read()
                if ret:
                    frame = cv2.resize(frame, (self.capture_width, self.capture_height))
                    self.frames[camera_id] = frame
                time.sleep(0.033)  # ~30 FPS

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
        """Hiển thị tất cả khung hình camera với layout 2x2 grid."""
        while self.running:
            frames = [self.frames.get(cam_id, np.zeros((self.capture_height, self.capture_width, 3), dtype=np.uint8))
                      for cam_id in self.camera_ids]

            if len(frames) == 3:
                # Resize frames to fit into the display squares
                resized_frames = [cv2.resize(frame, (self.display_width, self.display_height)) for frame in frames]

                # Create an empty black frame for the fourth square
                empty_frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
                # Arrange frames in a 2x2 grid
                top_row = np.hstack((resized_frames[0], resized_frames[1]))
                bottom_row = np.hstack((resized_frames[2], empty_frame))
                combined_frame = np.vstack((top_row, bottom_row))
            else:
                combined_frame = None

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