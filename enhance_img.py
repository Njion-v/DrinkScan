import torch
print(torch.__version__)  # Kiểm tra phiên bản
print(torch.cuda.is_available())  # Kiểm tra GPU đã hoạt động chưa
print(torch.cuda.get_device_name(0))  # Xem tên GPU