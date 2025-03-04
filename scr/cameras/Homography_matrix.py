import cv2
import torch
import numpy as np
from ultralytics import YOLO

def load_yolo_model(weights_path, conf_threshold=0.3):
    model = YOLO(weights_path)
    model.conf = conf_threshold
    return model

def detect_objects(model, image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)[0]
    return results.boxes.xyxy.cpu().numpy()

def extract_sift_features(image):
    sift = cv2.SIFT_create(1000)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(desc1, desc2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return np.array([[m.queryIdx, m.trainIdx] for m in good_matches])

def compute_homography(kpts1, kpts2, matches):
    pts1 = np.float32([kpts1[m[0]].pt for m in matches])
    pts2 = np.float32([kpts2[m[1]].pt for m in matches])
    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    return H

def apply_homography(H, boxes):
    transformed_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        pts = np.array([[x1, y1], [x2, y2]], dtype=np.float32).reshape(-1, 1, 2)
        transformed_pts = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        transformed_boxes.append([*transformed_pts[0], *transformed_pts[1]])
    return np.array(transformed_boxes)

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    
    xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
    xi2, yi2 = min(x2, x2_p), min(y2, y2_p)
    intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def remove_duplicates(boxes1, boxes2, iou_threshold=0.5):
    filtered_boxes = []
    for box2 in boxes2:
        if all(compute_iou(box1, box2) < iou_threshold for box1 in boxes1):
            filtered_boxes.append(box2)
    return np.array(filtered_boxes)

def visualize_results(image, boxes, color, label):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main():
    weights_path = "./models/Yolov11-24.2/detect/train/weights/best.pt"
    model = load_yolo_model(weights_path)
    
    img1 = cv2.imread("test/test_imgs/camera_1_20250217_113550.jpg", cv2.IMREAD_COLOR)
    img2 = cv2.imread("test/test_imgs/camera_2_20250217_113550.jpg", cv2.IMREAD_COLOR)
    
    kpts1, desc1 = extract_sift_features(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
    kpts2, desc2 = extract_sift_features(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    
    matches = match_features(desc1, desc2)
    H = compute_homography(kpts1, kpts2, matches)
    
    preds1 = detect_objects(model, img1)
    preds2 = detect_objects(model, img2)
    
    transformed_boxes = apply_homography(H, preds1[:, :4])
    final_boxes = remove_duplicates(transformed_boxes, preds2[:, :4])
    
    visualize_results(img1, preds1[:, :4], (0, 0, 255), "Detected")
    visualize_results(img2, preds2[:, :4], (0, 255, 0), "Detected")
    visualize_results(img2, transformed_boxes, (255, 0, 0), "Warped")
    visualize_results(img2, final_boxes, (0, 255, 255), "Final")
    cv2.imshow("img1", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # stacked_images = np.hstack([img1, img2])
    # cv2.imshow("Multi-Camera Tracking", stacked_images)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
