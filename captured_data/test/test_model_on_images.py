import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load model YOLO
model = YOLO("detect/yolov8_drinkscan_allsides/weights/best.pt")

# Thư mục chứa ảnh
image_dir = "test/test_imgs/"
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Xử lý từng ảnh trong thư mục
for image_path in image_files:
    # Đọc ảnh
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển về RGB để hiển thị đúng màu

    # Nhận diện đối tượng
    results = model(image_path)

    # Vẽ bounding box lên ảnh
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Lấy tọa độ box
            label = result.names[int(box.cls[0])]  # Nhãn lớp
            confidence = box.conf[0].item()  # Độ tin cậy

            # Vẽ bounding box lên ảnh
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_rgb, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Hiển thị ảnh đã nhận diện
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title(f"Detected objects in {os.path.basename(image_path)}")
    plt.show()
