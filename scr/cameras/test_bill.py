from collections import defaultdict
from tabulate import tabulate

def filter_detections(raw_detections, threshold=0.1):
    """
    Lọc kết quả phát hiện với ngưỡng confidence >= threshold.
    
    Args:
        raw_detections (list of tuples): Danh sách chứa (label, confidence, quantity).
        threshold (float): Ngưỡng confidence để chấp nhận sản phẩm.
    
    Returns:
        dict: Dictionary chứa số lượng sản phẩm hợp lệ.
    """
    filtered_results = defaultdict(int)
    for label, confidence, quantity in raw_detections:
        if confidence >= threshold:
            filtered_results[label] += quantity
    return filtered_results

def match_and_combine_results(cam_results):
    """
    Kết hợp kết quả từ nhiều camera, lấy số lượng lớn nhất nếu cùng nhãn.
    """
    combined_results = {}
    for cam_result in cam_results:
        for label, quantity in cam_result.items():
            combined_results[label] = max(combined_results.get(label, 0), quantity)
    return combined_results

def count_total_products(combined_results):
    """
    Đếm tổng số chai và lon từ kết quả đã kết hợp.
    """
    total_bottles = combined_results.get("bottle", 0)
    total_cans = combined_results.get("can", 0)
    return total_bottles, total_cans

def generate_final_output(cam_results):
    """
    Kết hợp kết quả từ nhiều camera và kiểm tra tính hợp lệ.
    """
    combined_results = match_and_combine_results(cam_results)
    total_bottles, total_cans = count_total_products(combined_results)
    return combined_results, total_bottles, total_cans

def display_results_table(combined_results, total_bottles, total_cans):
    """
    Hiển thị kết quả dưới dạng bảng.
    """
    table = [["Label", "Quantity"]] + [[label, quantity] for label, quantity in combined_results.items()]
    print(tabulate(table, headers="firstrow", tablefmt="pretty"))
    print(f"Total Bottles: {total_bottles}, Total Cans: {total_cans}")

# 🔹 Giả lập kết quả nhận diện từ 3 ảnh (Camera 1, 2, 3)
detections_cam1 = [("bottle", 0.85, 3), ("can", 0.65, 5)]
detections_cam2 = [("revive_lemon_salt", 0.90, 1), ("revive_regular", 0.80, 1), ("vinh_hao_water", 0.80, 1)]
detections_cam3 = [("cocacola", 0.50, 3), ("pepsi", 0.72, 1), ("beer_tiger", 0.95, 1),("vinh_hao_water", 0.80, 1)]

# 🔹 Lọc kết quả dựa trên confidence >= 0.7
filtered_cam1 = filter_detections(detections_cam1)
filtered_cam2 = filter_detections(detections_cam2)
filtered_cam3 = filter_detections(detections_cam3)

# 🔹 Kết hợp kết quả từ các camera
final_output, total_bottles, total_cans = generate_final_output([filtered_cam1, filtered_cam2, filtered_cam3])
filtered_data = {k: v for k, v in final_output.items() if k not in ['bottle', 'can']}
print(total_bottles + total_cans)
print(sum(filtered_data.values()))
# 🔹 Hiển thị bảng kết quả
