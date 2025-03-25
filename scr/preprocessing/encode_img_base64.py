import base64

def encode_images_to_base64(image_paths):
    encoded_images = {}

    for path in image_paths:
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            encoded_images[path] = encoded_string

    return encoded_images

# Danh sách 3 ảnh
image_paths = ["./scr/preprocessing/yolov11s_camera_0_20250317_103555.jpg", "./scr/preprocessing/yolov11s_camera_1_20250317_103555.jpg", "./scr/preprocessing/yolov11s_camera_2_20250317_103541.jpg"]

# Encode ảnh
encoded_images = encode_images_to_base64(image_paths)

# In base64 của từng ảnh
for img_name, encoded in encoded_images.items():
    print(f"\n{img_name}:\n{encoded[:100]}...")  # In 100 ký tự đầu để kiểm tra
