
from ultralytics import YOLO 
import cv2
import matplotlib.pyplot as plt

model = YOLO(r"D:\AI_Progress\3D_Computer_Vision\scr\models\Yolov11-bread_beverage\bread_full_v1.pt")

image_path = r"D:\AI_Progress\3D_Computer_Vision\captured_data\test\test_imgs\0809_F2_0809_camera_2_20250118_120516.jpg"

# Đọc ảnh
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển về RGB để hiển thị đúng màu

# Nhận diện đối tượng trong ảnh
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
plt.show()
