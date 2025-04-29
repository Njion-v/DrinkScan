from flask import Flask, request, jsonify
import yaml
from scr.drink_model import DrinkModel
from scr.utils import match_and_combine_results, count_total_products, check_totals, decode_base64_image

app = Flask(__name__)

with open("config/drink_model.yaml", "r") as f:
    drink_cfg = yaml.safe_load(f)
drink_model = DrinkModel(drink_cfg)


@app.route('/process_drink', methods=['POST'])
def process_drink():
    data = request.json
    cam_results = []

    for cam_id in ['camera1', 'camera2', 'camera3']:
        if cam_id not in data:
            return jsonify({"error": f"Thiếu ảnh từ {cam_id}"}), 400
        try:
            image = decode_base64_image(data[cam_id])
            cam_results.append(drink_model.infer(image))
        except Exception as e:
            return jsonify({"error": f"Lỗi với {cam_id}: {str(e)}"}), 400

    combined = match_and_combine_results(cam_results)
    bottle, can = count_total_products(combined)
    check_totals(combined, bottle, can)

    return jsonify({
        "total_products": bottle + can,
        "combined_results": {k: v for k, v in combined.items() if k not in ['bottle', 'can']}
    })

if __name__ == '__main__':
    app.run(debug=True)
