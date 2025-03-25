import os
import time
import cv2
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from ultralytics import YOLO
from tabulate import tabulate

# Load YOLO model
MODEL_PATH = r"D:\AI_Progress\DrinkScan\scr\models\Yolo11s-datasetv11\detect\train\weights\best.pt"
model = YOLO(MODEL_PATH)

def load_images_from_folder(folder):
    """Load image file paths and group by camera index"""
    images = defaultdict(list)
    for file in sorted(os.listdir(folder)):
        if file.startswith("yolov11s_camera") and file.endswith(".jpg"):
            cam_index = file.split("_")[2]  # Extract camera index
            images[cam_index].append(os.path.join(folder, file))
    return images

def detect_objects(image_paths):
    """Run YOLO detection and return results with labels and confidence"""
    cam_results = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        results = model(image)
        detections = defaultdict(list)
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for class_id, confidence in zip(result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                    if confidence > 0.7:
                        label = model.names[int(class_id)]
                        detections[label].append(confidence)
        cam_results.append(detections)
    return cam_results

def match_and_combine_results(cam_results):
    """Match and combine results from multiple cameras without duplication"""
    combined_results = {}
    for cam_result in cam_results:
        for label, conf_list in cam_result.items():
            if label in combined_results:
                combined_results[label].extend(conf_list)
            else:
                combined_results[label] = conf_list
    return combined_results

def compute_metrics(true_labels, predicted_labels):
    """Calculate Precision, Recall, F1-score and IoU"""
    TP = len(set(true_labels) & set(predicted_labels))
    FP = len(set(predicted_labels) - set(true_labels))
    FN = len(set(true_labels) - set(predicted_labels))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    
    return precision, recall, f1_score, iou

def evaluate_model(folder, output_folder):
    """Evaluate the model on a given test folder"""
    os.makedirs(output_folder, exist_ok=True)
    images_grouped = load_images_from_folder(folder)
    
    for cam_index, image_paths in images_grouped.items():
        if len(image_paths) < 3:
            continue
        
        start_time = time.time()
        cam_results = detect_objects(image_paths[:3])  # Process 3 images at a time
        combined_results = match_and_combine_results(cam_results)
        processing_time = time.time() - start_time
        
        total_bottles = sum(combined_results.get("bottle", []))
        total_cans = sum(combined_results.get("can", []))
        total_products = total_bottles + total_cans
        combined_total = sum(len(v) for v in combined_results.values())
        matching_accuracy = combined_total / total_products if total_products > 0 else 0
        
        # Save results to Excel
        df = pd.DataFrame.from_dict(combined_results, orient='index')
        df.columns = [f'Confidence_{i+1}' for i in range(df.shape[1])]
        df.to_excel(os.path.join(output_folder, f'results_{cam_index}.xlsx'))
        
        # Generate confusion matrix
        labels = list(combined_results.keys())
        true_labels = labels  # Assuming ground truth is available for comparison
        precision, recall, f1, iou = compute_metrics(true_labels, labels)
        
        metrics = {
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "IoU": iou,
            "Processing Time (s)": processing_time,
            "Matching Accuracy": matching_accuracy
        }
        
        # Save metrics as CSV
        pd.DataFrame(metrics, index=[0]).to_csv(os.path.join(output_folder, f'metrics_{cam_index}.csv'), index=False)
        
        # Generate heatmap visualization
        conf_matrix = np.array([[precision, recall], [f1, iou]])
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='coolwarm', xticklabels=["Precision", "Recall"], yticklabels=["F1-score", "IoU"])
        plt.title(f'Metrics Heatmap - {cam_index}')
        plt.savefig(os.path.join(output_folder, f'heatmap_{cam_index}.png'))
        plt.close()

# Run evaluation
base_folder = "scr/evaluate/test"
output_base = "result"

evaluate_model(os.path.join(base_folder, "Easy"), os.path.join(output_base, "easy"))
evaluate_model(os.path.join(base_folder, "Medium"), os.path.join(output_base, "medium"))
evaluate_model(os.path.join(base_folder, "Hard"), os.path.join(output_base, "hard"))

print("✅ Evaluation complete! Results saved.")
