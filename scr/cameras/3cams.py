import cv2
import numpy as np
import os
import threading
import time
from datetime import datetime
import torch
from ultralytics import YOLO
from extract_bill import generate_final_output, display_results_table,count_total_products  # Import functions
from collections import defaultdict

class MultiCameraYOLO:
    def __init__(self, camera_ids=[0, 1, 2]):
        self.camera_ids = camera_ids
        self.cameras = {}
        self.frames = {}
        self.running = True
        self.capture_width = 1920  # Capture resolution
        self.capture_height = 1080
        self.display_width = 640  # Display resolution for each square
        self.display_height = 400
        self.output_dir = "captured_images"
        os.makedirs(self.output_dir, exist_ok=True)

        # Load YOLOv11 model
        self.model = YOLO(r"D:\AI_Progress\DrinkScan\scr\models\Yolo11s-datasetv11\detect\train\weights\best.pt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
        self.model.to(self.device)

        # Initialize cameras
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
        """
        Capture images from all active cameras with bounding boxes drawn.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for cam_id, frame in self.frames.items():
            if frame is not None:
                # # Perform YOLO inference on the frame
                # result = self.model(frame, device=self.device)
                # if result:  # If there are detection results
                #     # Draw bounding boxes on the frame
                #     frame_with_boxes = self._draw_bounding_boxes(frame, result)
                # else:
                #     # If no detections, use the original frame
                #     frame_with_boxes = frame

                # Save the frame with bounding boxes
                filename = f"{self.output_dir}/yolov11s_camera_{cam_id}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
        print(f"Ảnh được chụp vào {timestamp}")

    def _draw_bounding_boxes(self, frame, results):
        """Draw bounding boxes on the frame using YOLO detection results."""
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes in (x1, y1, x2, y2) format
            confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
            class_ids = result.boxes.cls.cpu().numpy()  # Get class IDs

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                if confidence > 0.7:
                    x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
                    label = f"{self.model.names[int(class_id)]} {confidence:.2f}"  # Create label

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame
    def create_results_table_image(self, final_output, total_products, combined_quantities):
        """
        Create an image displaying the results table and warning (if any).
        Args:
            final_output (dict): Dictionary containing labels and quantities.
            total_products (int): Total number of products (bottles + cans).
            combined_quantities (int): Sum of quantities in final_output.
        Returns:
            np.ndarray: Image containing the results table.
        """
        # Create a blank image
        table_image = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)

        # Define text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)  # White color
        line_type = 2

        # Add warning if quantities do not match
        if total_products != combined_quantities:
            warning_text = f"Warning: Quantities do not match!"
            cv2.putText(table_image, warning_text, (10, 30), font, font_scale, (0, 0, 255), line_type)  # Red color for warning

        # Add table header
        total = f"Total Products: {total_products}, Combined quantites: {combined_quantities}"
        cv2.putText(table_image, total, (10, 50), font, font_scale, (0, 255,0), line_type)  
        cv2.putText(table_image, "+-------------------+----------+", (10, 70), font, font_scale, font_color, line_type)
        cv2.putText(table_image, "|       Label       | Quantity |", (10, 100), font, font_scale, font_color, line_type)
        cv2.putText(table_image, "+-------------------+----------+", (10, 130), font, font_scale, font_color, line_type)

        # Add table rows
        y_offset = 160
        for label, quantity in final_output.items():
            row_text = f"| {label:<17} | {quantity:>8} |"
            cv2.putText(table_image, row_text, (10, y_offset), font, font_scale, font_color, line_type)
            y_offset += 30

        # Add table footer
        cv2.putText(table_image, "+-------------------+----------+", (10, y_offset), font, font_scale, font_color, line_type)

        return table_image
    def run(self):
        """Hiển thị tất cả khung hình camera với layout linh hoạt và bảng kết quả."""
        while self.running:
            frames = []
            results = []
            for cam_id in self.camera_ids:
                if cam_id in self.cameras:  # Kiểm tra xem camera có hoạt động không
                    frame = self.frames.get(cam_id)
                    if frame is not None:
                        result = self.model(frame, device=self.device)
                        if result:  # Kiểm tra xem có kết quả nào không
                            results.append(result[0])  # Lấy phần tử đầu tiên của danh sách
                            frames.append(frame)

            if frames:
                print(f"Số lượng camera hoạt động: {len(frames)}")
                print(f"Số lượng kết quả từ YOLO: {len(results)}")

                # Extract detection results for each active camera
                cam_results = []
                for result in results:
                    detections = defaultdict(int)
                    if hasattr(result, 'boxes') and result.boxes is not None:  # Kiểm tra xem result có thuộc tính boxes không
                        for class_id,confidence  in zip(result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                            if confidence > 0.7:
                                label = self.model.names[int(class_id)]
                                detections[label] += 1
                    cam_results.append(detections)

                # Generate final output
                final_output = generate_final_output(cam_results)

                # Count total products (bottles + cans)
                total_bottles, total_cans = count_total_products(final_output)
                total_products = total_bottles + total_cans

                # Calculate combined quantities
                beverage_only = {k: v for k, v in final_output.items() if k not in ['bottle', 'can']}
                combined_quantities = sum(beverage_only.values())

                # Display the results table
                display_results_table(final_output)

                # Draw bounding boxes on each active frame
                frames_with_boxes = []
                for frame, result in zip(frames, results):
                    frame_with_boxes = frame  # Khởi tạo frame_with_boxes với frame gốc
                    if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:  # Nếu có đối tượng được phát hiện
                        frame_with_boxes = self._draw_bounding_boxes(frame, result)
                    # Resize frame về kích thước hiển thị
                    frame_with_boxes = cv2.resize(frame_with_boxes, (self.display_width, self.display_height))
                    frames_with_boxes.append(frame_with_boxes)

                # Create results table image
                table_image = self.create_results_table_image(final_output, total_products, combined_quantities)

                # Resize frames to fit into the display layout
                if len(frames) == 1:
                    combined_frame = frames_with_boxes[0]
                elif len(frames) == 2:
                    top_row = np.hstack((frames_with_boxes[0], frames_with_boxes[1]))
                    combined_frame = np.vstack((top_row, np.zeros((self.display_height, self.display_width * 2, 3), dtype=np.uint8)))
                else:
                    top_row = np.hstack((frames_with_boxes[0], frames_with_boxes[1]))
                    bottom_row = np.hstack((frames_with_boxes[2], table_image))
                    combined_frame = np.vstack((top_row, bottom_row))

                # Hiển thị khung hình
                if combined_frame is not None and combined_frame.size != 0:
                    cv2.imshow("Multi-Camera YOLO Detection", combined_frame)
                else:
                    print("Khung hình không hợp lệ hoặc rỗng.")

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