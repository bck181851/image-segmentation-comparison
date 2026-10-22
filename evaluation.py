import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from sklearn.cluster import KMeans
from skimage import segmentation, measure, filters, color
from ultralytics import FastSAM
from scipy.io import loadmat
from scipy.ndimage import distance_transform_edt
import pandas as pd

image_dir = Path("BSR/BSDS500/data/images/train")
gt_dir = Path("BSR/BSDS500/data/groundTruth/train")
output_dir = Path("results")

def majority_vote(predicted, ground_truth):
    matched = np.zeros_like(ground_truth)
    for region_id in np.unique(predicted):
        mask = predicted == region_id 
        gt_labels, counts = np.unique(ground_truth[mask], return_counts = True)
        majority_label = gt_labels[np.argmax(counts)]
        matched[mask] = majority_label
    return matched  

def calculate_metrics(matched, ground_truth):
    ious, dices = list(), list()
    for region_id in np.unique(ground_truth):
        pred_mask =  matched == region_id 
        gt_mask = ground_truth == region_id 
        intersection = np.sum(pred_mask & gt_mask)
        union = np.sum(pred_mask | gt_mask) 
        iou = intersection / union 
        dice = 2*intersection / (np.sum(pred_mask) + np.sum(gt_mask))
        ious += [iou]
        dices += [dice]
    pixel_accuracy = np.mean(matched == ground_truth)
    return pixel_accuracy, np.mean(ious), np.mean(dices) 

def boundary_fscore(predicted, ground_truth_boundary, tolerance = 2):
    pred_boundary = segmentation.find_boundaries(predicted)
    gt_boundary = ground_truth_boundary.astype(bool)
    if not pred_boundary.any() and not gt_boundary.any():
        return 1.0 
    if not pred_boundary.any() or not gt_boundary.any():
        return 0.0 
    dist_to_gt = distance_transform_edt(~gt_boundary)
    dist_to_pred = distance_transform_edt(~pred_boundary)
    precision = np.mean(dist_to_gt[pred_boundary] <= tolerance)
    recall = np.mean(dist_to_pred[gt_boundary] <= tolerance)
    if precision + recall == 0: return 0.0
    return 2*precision*recall / (precision + recall)

def evaluate_image(predicted, gt_path):
    data = loadmat(gt_path)
    annotations = data["groundTruth"][0]
    results = list()
    for gt in annotations:
        gt_labels = gt["Segmentation"][0, 0]
        gt_boundaries = gt["Boundaries"][0, 0]
        matched = majority_vote(predicted, gt_labels)
        accuracy, miou, mdice = calculate_metrics(matched, gt_labels)
        fscore = boundary_fscore(predicted, gt_boundaries)
        results += [[accuracy, miou, mdice, fscore]]
    return np.mean(results, axis = 0)

methods = {
    "K-means k=2": "kmeans_k2_regions.npy",
    "K-means k=3": "kmeans_k3_regions.npy",
    "K-means k=4": "kmeans_k4_regions.npy",
    "K-means k=6": "kmeans_k6_regions.npy",
    "K-means k=8": "kmeans_k8_regions.npy",
    "Watershed": "watershed_labels.npy",
    "FastSAM": "fastsam_labels.npy"
}

results = list()

for path in sorted(image_dir.glob("*.jpg"))[:30]:
    gt_path = gt_dir / f"{path.stem}.mat"
    save_dir = output_dir / path.stem 
    for method, filename in methods.items():
        predicted = np.load(save_dir / filename)
        accuracy, miou, mdice, fscore = evaluate_image(predicted, gt_path)
        num_regions = len(np.unique(predicted))
        cur = {
            "Image" : path.name, 
            "Method": method, 
            "Pixel Accuracy": accuracy, 
            "Mean IoU": miou,
            "Mean Dice": mdice, 
            "Boundary F-score": fscore, 
            "Number of Regions": num_regions
        }
        results.append(cur)
        print(f"results for {path.stem}, {filename} appended")

report_dir = output_dir / "report"
report_dir.mkdir(parents = True, exist_ok = True)
df = pd.DataFrame(results)
df.to_csv(report_dir / "metrics_by_image.csv", index = False)

summary = df.groupby("Method").agg({
    "Pixel Accuracy" : "mean",
    "Mean IoU": "mean",
    "Mean Dice": "mean",
    "Boundary F-score": "mean",
    "Number of Regions": "mean"
})

summary = summary.round(4)
summary.to_csv(report_dir / "method_summary.csv")

print("Evalution results saved successfully")
