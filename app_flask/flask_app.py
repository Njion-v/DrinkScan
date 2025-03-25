from flask import Flask, request, jsonify
import base64
from PIL import Image
from io import BytesIO
from tabulate import tabulate
from ultralytics import YOLO  # Import YOLO từ Ultralytics

app = Flask(__name__)

# Load YOLO model
model = YOLO(r"D:\AI_Progress\DrinkScan\scr\models\Yolo11s-datasetv11\detect\train\weights\best.pt")

def apply_ai_model(image):
    """
    Xử lý ảnh bằng YOLO model và trả về dictionary chứa nhãn và số lượng.
    """
    # Chạy YOLO model trên ảnh
    results = model(image)

    # Phân tích kết quả để đếm các đối tượng được phát hiện
    detected_objects = {}
    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls)]  # Lấy tên nhãn
            if label in detected_objects:
                detected_objects[label] += 1
            else:
                detected_objects[label] = 1

    return detected_objects

# Helper functions (được cung cấp trong câu hỏi)
def match_and_combine_results(cam_results):
    combined_results = {}
    for cam_result in cam_results:
        for label, quantity in cam_result.items():
            if label in combined_results:
                combined_results[label] = max(combined_results[label], quantity)
            else:
                combined_results[label] = quantity
    return combined_results

def count_total_products(combined_results):
    total_bottles = combined_results.get("bottle", 0)
    total_cans = combined_results.get("can", 0)
    return total_bottles, total_cans

def check_quantities(combined_results, total_bottles, total_cans):
    total_products = total_bottles + total_cans
    total_combined = sum(combined_results.values())
    if total_products != total_combined:
        print(f"Warning: Quantities do not match! Total products: {total_products}, Combined quantities: {total_combined}")
        return False
    return True

def generate_final_output(cam_results):
    combined_results = match_and_combine_results(cam_results)
    total_bottles, total_cans = count_total_products(combined_results)
    check_quantities(combined_results, total_bottles, total_cans)
    return combined_results

def display_results_table(combined_results):
    table = []
    for label, quantity in combined_results.items():
        table.append([label, quantity])
    print(tabulate(table, headers=["Label", "Quantity"], tablefmt="pretty"))

# Flask route để xử lý ảnh
@app.route('/process_images', methods=['POST'])
def process_images():
    # Lấy dữ liệu JSON từ request
    data = request.json
    if not data or 'cam_left' not in data or 'cam_right' not in data or 'cam_top' not in data:
        return jsonify({"error": "Vui lòng cung cấp đủ 3 ảnh: cam_left, cam_right, cam_top."}), 400

    # Danh sách chứa kết quả từ các camera
    cam_results = []

    # Xử lý từng ảnh
    for cam_key in ['cam_left', 'cam_right', 'cam_top']:
        try:
            # Giải mã base64 và chuyển thành ảnh
            img_data = base64.b64decode(data[cam_key])
            image = Image.open(BytesIO(img_data))

            # Áp dụng AI model để xử lý ảnh
            result = apply_ai_model(image)
            cam_results.append(result)
        except Exception as e:
            return jsonify({"error": f"Lỗi khi xử lý ảnh từ {cam_key}: {str(e)}"}), 400

    # Tạo kết quả cuối cùng
    combined_results = generate_final_output(cam_results)
    total_bottles, total_cans = count_total_products(combined_results)
    total_products = total_bottles + total_cans
    # Hiển thị kết quả dưới dạng bảng
    display_results_table(combined_results)
    beverage_only = {k: v for k, v in combined_results.items() if k not in ['bottle', 'can']}
    # Chuẩn bị phản hồi
    response = {
        "total_products": total_products,
        "combined_results": beverage_only
    }

    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True)