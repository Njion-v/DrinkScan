import base64
import requests

def encode_images_to_base64(image_paths):
    encoded_images = {}

    for path in image_paths:
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            encoded_images[path] = encoded_string

    return encoded_images

# Danh sách 3 ảnh
image_paths = [
    "./scr/preprocessing/yolov11s_camera_0_20250317_103555.jpg",
    "./scr/preprocessing/yolov11s_camera_1_20250317_103555.jpg",
    "./scr/preprocessing/yolov11s_camera_2_20250317_103541.jpg"
]

# Encode ảnh
encoded_images = encode_images_to_base64(image_paths)

# Gửi API Flask (chạy trên localhost)
api_url = "http://127.0.0.1:5000/upload"

# Chuẩn bị payload JSON
data = {
    "cam_left": encoded_images[image_paths[0]],
    "cam_right": encoded_images[image_paths[1]],
    "cam_top": encoded_images[image_paths[2]]
}

# Gửi request POST đến Flask API
response = requests.post(api_url, json=data)

# Kiểm tra phản hồi từ API
print("Response Status Code:", response.status_code)
print("Response JSON:", response.json())  # Xem phản hồi từ Flask
