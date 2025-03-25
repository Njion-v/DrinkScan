import cv2
import numpy as np
import os
import threading
import time
from datetime import datetime
import torch
from ultralytics import YOLO
from extract_bill import generate_final_output, display_results_table, count_total_products
from collections import defaultdict

class MultiCameraYOLO:
    def __init__(self, camera_ids=[0, 1, 2]):
        self.camera_ids = camera_ids
        self.cameras = {}
        self.frames = {}
        self.running = True
        self.capture_width = 1920
        self.capture_height = 1080
        self.display_width = 640
        self.display_height = 400
        self.output_dir = "captured_images"
        os.makedirs(self.output_dir, exist_ok=True)

        # Model YOLO
        self.model = YOLO("best.pt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Ghi hình
        self.video_writers = {}
        self.recording_single = False
        self.recording_all = False
        self.all_video_writer = None
        self.all_video_filename = "output/all_cameras.avi"

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
                print(f"Không thể mở camera {cam_id}")

    def _capture_thread(self, camera_id):
        while self.running:
            if camera_id in self.cameras:
                ret, frame = self.cameras[camera_id].read()
                if ret:
                    frame = cv2.resize(frame, (self.capture_width, self.capture_height))
                    self.frames[camera_id] = frame
                    if self.recording_single and camera_id in self.video_writers:
                        self.video_writers[camera_id].write(frame)
                time.sleep(0.033)

    def _start_capture_threads(self):
        for cam_id in self.cameras.keys():
            thread = threading.Thread(target=self._capture_thread, args=(cam_id,))
            thread.daemon = True
            thread.start()
            self.capture_threads.append(thread)

    def _toggle_recording_single(self):
        if self.recording_single:
            print("Dừng ghi hình từng camera")
            for writer in self.video_writers.values():
                writer.release()
            self.video_writers.clear()
        else:
            print("Bắt đầu ghi hình từng camera")
            for cam_id in self.cameras:
                filename = f"output/camera_{cam_id}.avi"
                self.video_writers[cam_id] = cv2.VideoWriter(
                    filename, cv2.VideoWriter_fourcc(*"XVID"), 30, (self.capture_width, self.capture_height)
                )
        self.recording_single = not self.recording_single

    def _toggle_recording_all(self, combined_frame):
        if self.recording_all:
            print("Dừng ghi hình toàn bộ hệ thống")
            if self.all_video_writer:
                self.all_video_writer.release()
                self.all_video_writer = None
        else:
            print("Bắt đầu ghi hình toàn bộ hệ thống")
            self.all_video_writer = cv2.VideoWriter(
                self.all_video_filename, cv2.VideoWriter_fourcc(*"XVID"), 30,
                (combined_frame.shape[1], combined_frame.shape[0])
            )
        self.recording_all = not self.recording_all

    def run(self):
        while self.running:
            frames = [self.frames[cam_id] for cam_id in self.camera_ids if cam_id in self.frames]
            if len(frames) == 3:
                top_row = np.hstack((frames[0], frames[1]))
                bottom_row = np.hstack((frames[2], np.zeros((self.capture_height, self.capture_width, 3), dtype=np.uint8)))
                combined_frame = np.vstack((top_row, bottom_row))
            else:
                combined_frame = frames[0] if frames else None

            if combined_frame is not None:
                cv2.imshow("Multi-Camera View", combined_frame)
                if self.recording_all and self.all_video_writer:
                    self.all_video_writer.write(combined_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('c'):
                self._toggle_recording_single()
            elif key == ord('x'):
                self._toggle_recording_all(combined_frame)

        self._cleanup()

    def _cleanup(self):
        self.running = False
        for cap in self.cameras.values():
            cap.release()
        for writer in self.video_writers.values():
            writer.release()
        if self.all_video_writer:
            self.all_video_writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    capture_system = MultiCameraYOLO()
    capture_system.run()
