import cv2
import numpy as np
import os
import threading
import time
from datetime import datetime
import torch
from ultralytics import YOLO
from collections import defaultdict
from matching import generate_final_output, display_results_table, count_total_products

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
        self.model = self._load_model()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.capture_threads = []
        self._setup()

    def _load_model(self):
        model = YOLO(r"D:\AI_Progress\DrinkScan\checkpoints\Yolov11s-v15\detect\train\weights\best.pt")
        return model.to(self.device)

    def _setup(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self._init_cameras()
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
                print(f"Warning: Could not open camera {cam_id}")

    def _capture_thread(self, camera_id):
        while self.running and camera_id in self.cameras:
            ret, frame = self.cameras[camera_id].read()
            if ret:
                frame = cv2.resize(frame, (self.capture_width, self.capture_height))
                self.frames[camera_id] = frame
            time.sleep(0.033)

    def _start_capture_threads(self):
        for cam_id in self.cameras:
            thread = threading.Thread(target=self._capture_thread, args=(cam_id,))
            thread.daemon = True
            thread.start()
            self.capture_threads.append(thread)

    def _draw_bounding_boxes(self, frame, result):
        for box, confidence, class_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy()
        ):
            if confidence > 0.7:
                x1, y1, x2, y2 = map(int, box)
                label = f"{self.model.names[int(class_id)]} {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

    def capture_images(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for cam_id, frame in self.frames.items():
            if frame is not None:
                filename = f"{self.output_dir}/yolov11s_camera_{cam_id}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
        print(f"Images captured at {timestamp}")

    def create_results_table_image(self, final_output, total_products, combined_quantities):
        table_image = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)
        line_type = 2

        if total_products != combined_quantities:
            cv2.putText(table_image, "Warning: Quantities do not match!", (10, 30), font, font_scale, (0, 0, 255), line_type)

        total = f"Total Products: {total_products}, Combined quantities: {combined_quantities}"
        cv2.putText(table_image, total, (10, 50), font, font_scale, (0, 255, 0), line_type)
        cv2.putText(table_image, "+-------------------+----------+", (10, 70), font, font_scale, font_color, line_type)
        cv2.putText(table_image, "|       Label       | Quantity |", (10, 100), font, font_scale, font_color, line_type)
        cv2.putText(table_image, "+-------------------+----------+", (10, 130), font, font_scale, font_color, line_type)

        y_offset = 160
        for label, quantity in final_output.items():
            row_text = f"| {label:<17} | {quantity:>8} |"
            cv2.putText(table_image, row_text, (10, y_offset), font, font_scale, font_color, line_type)
            y_offset += 30

        cv2.putText(table_image, "+-------------------+----------+", (10, y_offset), font, font_scale, font_color, line_type)
        return table_image

    def run(self):
        while self.running:
            frames = []
            results = []
            for cam_id in self.camera_ids:
                if cam_id in self.cameras and self.frames.get(cam_id) is not None:
                    frame = self.frames[cam_id]
                    result = self.model(frame, device=self.device)
                    if result:
                        frames.append(frame)
                        results.append(result[0])

            if frames:
                print(f"Active cameras: {len(frames)}")
                print(f"YOLO results: {len(results)}")

                cam_results = []
                for result in results:
                    detections = defaultdict(int)
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        for class_id, confidence in zip(result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                            if confidence > 0.7:
                                detections[self.model.names[int(class_id)]] += 1
                    cam_results.append(detections)

                final_output = generate_final_output(cam_results)
                total_bottles, total_cans = count_total_products(final_output)
                total_products = total_bottles + total_cans
                beverage_only = {k: v for k, v in final_output.items() if k not in ['bottle', 'can']}
                combined_quantities = sum(beverage_only.values())
                display_results_table(final_output)

                frames_with_boxes = []
                for frame, result in zip(frames, results):
                    frame_with_boxes = frame
                    if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                        frame_with_boxes = self._draw_bounding_boxes(frame, result)
                    frame_with_boxes = cv2.resize(frame_with_boxes, (self.display_width, self.display_height))
                    frames_with_boxes.append(frame_with_boxes)

                table_image = self.create_results_table_image(final_output, total_products, combined_quantities)

                if len(frames) == 1:
                    combined_frame = frames_with_boxes[0]
                elif len(frames) == 2:
                    top_row = np.hstack((frames_with_boxes[0], frames_with_boxes[1]))
                    combined_frame = np.vstack((top_row, np.zeros((self.display_height, self.display_width * 2, 3), dtype=np.uint8)))
                else:
                    top_row = np.hstack((frames_with_boxes[0], frames_with_boxes[1]))
                    bottom_row = np.hstack((frames_with_boxes[2], table_image))
                    combined_frame = np.vstack((top_row, bottom_row))

                if combined_frame is not None and combined_frame.size != 0:
                    cv2.imshow("Multi-Camera YOLO Detection", combined_frame)
                else:
                    print("Invalid or empty frame")

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