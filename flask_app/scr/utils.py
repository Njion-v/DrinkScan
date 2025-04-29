import base64
from PIL import Image
from io import BytesIO

def match_and_combine_results(cam_results):
    combined = {}
    for result in cam_results:
        for label, count in result.items():
            combined[label] = max(combined.get(label, 0), count)
    return combined

def count_total_products(results):
    return results.get("bottle", 0), results.get("can", 0)

def check_totals(results, bottles, cans):
    if bottles + cans != sum(results.values()):
        print("⚠ Warning: Mismatch in total quantities")

def decode_base64_image(encoded_str):
    try:
        # Loại bỏ phần tiền tố "data:image/jpeg;base64,"
        if "," in encoded_str:
            base64_data = encoded_str.split(",")[1]
        else:
            base64_data = encoded_str
        image_data = base64.b64decode(base64_data)
        return Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Không thể decode ảnh base64: {str(e)}")
