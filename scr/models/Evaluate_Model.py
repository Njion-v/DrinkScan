import os
import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load model YOLO
model = YOLO("detect/yolov8_drinkscan_allsides/weights/best.pt")

# Kiểm tra xem model có chạy trên GPU không
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Đánh giá trên tập validation
metrics = model.val()

# Trích xuất các chỉ số quan trọng
map50 = metrics.box.map50  # mAP@50
map50_95 = metrics.box.map  # mAP@50-95
precision = metrics.box.precision  # Precision
recall = metrics.box.recall  # Recall

print("\n🎯 **Evaluation Metrics:**")
print(f"✅ mAP@50: {map50:.4f}")
print(f"✅ mAP@50-95: {map50_95:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")

# Đọc lịch sử Loss từ quá trình huấn luyện
loss_file = "detect/yolov8_drinkscan_allsides/train/results.csv"

if os.path.exists(loss_file):
    import pandas as pd
    df = pd.read_csv(loss_file)

    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss", color="red")
    plt.plot(df["epoch"], df["val/box_loss"], label="Val Box Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Training")
    plt.legend()
    plt.show()
else:
    print("\n⚠️ Không tìm thấy file kết quả huấn luyện để vẽ Loss.")

# Đánh giá thực tế trên tập test
image_dir = "test/test_imgs/"
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

for image_path in image_files:
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển về RGB

    results = model(image_path)

    # Vẽ bounding box lên ảnh
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            confidence = box.conf[0].item()

            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_rgb, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Hiển thị kết quả nhận diện
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title(f"Detected objects in {os.path.basename(image_path)}")
    plt.show()
