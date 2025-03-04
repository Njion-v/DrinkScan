import cv2
import torch
from ultralytics import YOLO
import time
import os



# Cấu hình OpenCV để tránh lỗi trên Windows
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# Tạo thư mục lưu ảnh nếu chưa có
save_dir = "detections"
os.makedirs(save_dir, exist_ok=True)

# Danh sách camera cần sử dụng
camera_ids = [0, 1, 2]

# Mở tất cả camera một lần
caps = {cam_id: cv2.VideoCapture(cam_id) for cam_id in camera_ids}

# Kiểm tra xem có camera nào không mở được không
for cam_id in list(caps.keys()):
    if not caps[cam_id].isOpened():
        print(f"❌ Không thể mở camera {cam_id}")
        caps.pop(cam_id)  # Loại bỏ camera bị lỗi

# Load model YOLO lên GPU nếu khả dụng
model = YOLO("models/Yolov11-24.2/detect/train/weights/best.pt")

while True:
    frames = {}
    detected = {}

    # Đọc frame từ tất cả camera
    for cam_id, cap in caps.items():
        ret, frame = cap.read()
        if ret:
            frames[cam_id] = frame
            detected[cam_id] = False  # Ban đầu chưa phát hiện gì

    # Chạy YOLO trên tất cả frame nếu có dữ liệu
    if frames:
        batch_frames = list(frames.values())

        # Dự đoán trên GPU nếu có

        for cam_id, (frame, results) in zip(frames.keys(), zip(batch_frames, batch_results)):
            # Vẽ bounding box và kiểm tra đối tượng
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = result.names[int(box.cls[0])]
                    confidence = box.conf[0].item()

                    # Vẽ bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    detected[cam_id] = True  # Đánh dấu phát hiện

            # Nếu có phát hiện, lưu ảnh
            if detected[cam_id]:
                timestamp = int(time.time())
                img_path = os.path.join(save_dir, f"camera_{cam_id}_{timestamp}.jpg")
                cv2.imwrite(img_path, frame)
                print(f"📸 Ảnh chụp từ Camera {cam_id} đã lưu: {img_path}")

            # Hiển thị video
            cv2.imshow(f"Camera {cam_id}", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
for cap in caps.values():
    cap.release()

cv2.destroyAllWindows()
