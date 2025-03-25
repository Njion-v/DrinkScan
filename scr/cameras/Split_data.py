import os
import shutil

# Đường dẫn đến thư mục chứa ảnh
source_folder = "captured_images_new"

# Các thư mục đích
destination_folders = {
    "camera_0": "cam_0",
    "camera_1": "cam_1",
    "camera_3": "cam_3", 
}
# Tạo các thư mục đích nếu chưa tồn tại
for folder in destination_folders.values():
    os.makedirs(folder, exist_ok=True)

# Duyệt qua tất cả các tệp trong thư mục gốc
for file_name in os.listdir(source_folder):
    # Kiểm tra xem tệp có bắt đầu bằng các tiền tố chỉ định không
    for prefix, dest_folder in destination_folders.items():
        if file_name.startswith(prefix):
            # Đường dẫn tệp gốc và đích
            source_path = os.path.join(source_folder, file_name)
            dest_path = os.path.join(dest_folder, file_name)
            # Di chuyển tệp
            shutil.move(source_path, dest_path)
            print(f"Moved: {file_name} -> {dest_folder}")
            break


        
