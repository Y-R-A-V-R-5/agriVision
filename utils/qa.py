# Standard Libraries
import os
import sys
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

# Image Processing
import cv2
import imghdr
import numpy as np
from PIL import Image, ImageStat
import imagehash

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Clustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# For reading YAML if needed
import yaml


# ---------------------------
# Helper Functions
# ---------------------------

# ---------------------------
# Helper Functions for QA.py
# ---------------------------

# ---------------------------
# QA Helper: Simple Bar Plot for Counts
# ---------------------------

def plot_bar_simple(categories, counts, title="", xlabel="", ylabel="Count"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 5))
    sns.barplot(x=categories, y=counts, palette="viridis")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


# ---------------------------
# QA Helper: Histogram Plot for Distributions
# ---------------------------

def plot_histogram(data, bins=30, title="", xlabel="", ylabel="Frequency"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 5))
    sns.histplot(data, bins=bins, kde=True, color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


# ---------------------------
# QA Helper: Scatter Plot for Clustering Visuals
# ---------------------------

def plot_scatter(x, y, labels=None, title="", xlabel="", ylabel="", legend=True):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    plt.figure(figsize=(8, 6))
    if labels is not None:
        unique_labels = np.unique(labels)
        palette = sns.color_palette("hsv", len(unique_labels))
        for i, ul in enumerate(unique_labels):
            idx = labels == ul
            plt.scatter(x[idx], y[idx], label=str(ul), color=palette[i], alpha=0.6, edgecolor='k', s=50)
        if legend:
            plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(x, y, alpha=0.6, edgecolor='k', s=50)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


# ---------------------------
# QA Helper: Heatmap Plot for Overlap or Correlation
# ---------------------------

def plot_heatmap(matrix, xticklabels=None, yticklabels=None, title="", cmap="viridis"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, xticklabels=xticklabels, yticklabels=yticklabels, cmap=cmap, annot=True, fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def check_bbox_area_distribution(all_labels, class_names=None):
    """
    Plot the distribution of bounding box areas per class.
    """
    areas_by_class = defaultdict(list)
    for labels in all_labels:
        for label in labels:
            class_id, x_center, y_center, width, height = label
            area = width * height
            areas_by_class[class_id].append(area)

    if not any(areas_by_class.values()):
        print("No bounding boxes found for area distribution.")
        return

    plt.figure(figsize=(12, 6))
    for class_id, areas in areas_by_class.items():
        if len(areas) == 0:
            continue
        label = class_names[class_id] if class_names and class_id < len(class_names) else f"Class {class_id}"
        sns.kdeplot(areas, label=label, fill=True)

    plt.title("Bounding Box Area Distribution per Class")
    plt.xlabel("Normalized Bounding Box Area (width * height)")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


def check_bbox_aspect_ratios(all_labels, class_names=None):
    """
    Plot distribution of bounding box aspect ratios (width/height) per class.
    """
    aspect_ratios_by_class = defaultdict(list)
    for labels in all_labels:
        for label in labels:
            class_id, x_center, y_center, width, height = label
            if height == 0:
                continue
            aspect_ratio = width / height
            aspect_ratios_by_class[class_id].append(aspect_ratio)

    if not any(aspect_ratios_by_class.values()):
        print("No bounding boxes found for aspect ratio distribution.")
        return

    plt.figure(figsize=(12, 6))
    for class_id, ars in aspect_ratios_by_class.items():
        if len(ars) == 0:
            continue
        label = class_names[class_id] if class_names and class_id < len(class_names) else f"Class {class_id}"
        sns.kdeplot(ars, label=label, fill=True)

    plt.title("Bounding Box Aspect Ratio Distribution per Class")
    plt.xlabel("Aspect Ratio (width / height)")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


def check_bbox_overlap_per_image(all_labels, threshold=0.5):
    """
    Checks and reports images where bounding boxes overlap above a threshold.
    Overlaps are calculated by Intersection over Union (IoU).
    """
    def iou(box1, box2):
        # box: [x_center, y_center, width, height]
        x1_min = box1[0] - box1[2]/2
        x1_max = box1[0] + box1[2]/2
        y1_min = box1[1] - box1[3]/2
        y1_max = box1[1] + box1[3]/2

        x2_min = box2[0] - box2[2]/2
        x2_max = box2[0] + box2[2]/2
        y2_min = box2[1] - box2[3]/2
        y2_max = box2[1] + box2[3]/2

        inter_x_min = max(x1_min, x2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_min = max(y1_min, y2_min)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        if union_area == 0:
            return 0
        return inter_area / union_area

    overlapping_images = []
    for idx, labels in enumerate(tqdm(all_labels, desc="Checking bbox overlaps")):
        overlap_found = False
        n = len(labels)
        for i in range(n):
            for j in range(i+1, n):
                box1 = labels[i][1:]  # skip class_id
                box2 = labels[j][1:]
                if iou(box1, box2) > threshold:
                    overlap_found = True
                    break
            if overlap_found:
                break
        if overlap_found:
            overlapping_images.append(idx)

    if not overlapping_images:
        print("No images found with bounding box overlap greater than threshold.")
    else:
        print(f"{len(overlapping_images)} images found with bounding box overlap > {threshold}.")

    return overlapping_images


def check_class_imbalance(all_labels, class_names=None):
    """
    Calculate and plot class distribution and imbalance.
    """
    class_counts = Counter()
    total_labels = 0
    for labels in all_labels:
        for label in labels:
            class_id = int(label[0])
            class_counts[class_id] += 1
            total_labels += 1

    if total_labels == 0:
        print("No labeled bounding boxes found for class imbalance.")
        return

    class_ids = list(class_counts.keys())
    counts = [class_counts[cid] for cid in class_ids]
    labels = [class_names[cid] if class_names and cid < len(class_names) else f"Class {cid}" for cid in class_ids]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=labels, y=counts)
    plt.title("Class Distribution (imbalance check)")
    plt.ylabel("Number of bounding boxes")
    plt.xlabel("Class")
    plt.xticks(rotation=45, ha="right")
    plt.show()


def check_class_inconsistencies_across_splits(class_counts_train, class_counts_val, class_counts_test, class_names=None):
    """
    Compare class distributions across train, val, and test splits for inconsistencies.
    class_counts_* are Counter objects or dicts {class_id: count}.
    """
    import pandas as pd
    classes = set(class_counts_train) | set(class_counts_val) | set(class_counts_test)
    data = []
    for cid in classes:
        data.append({
            "class_id": cid,
            "train": class_counts_train.get(cid, 0),
            "val": class_counts_val.get(cid, 0),
            "test": class_counts_test.get(cid, 0),
            "class_name": class_names[cid] if class_names and cid < len(class_names) else f"Class {cid}"
        })
    df = pd.DataFrame(data).set_index("class_name")
    df.plot(kind='bar', figsize=(12, 6), rot=45)
    plt.title("Class Distribution Across Splits")
    plt.ylabel("Number of Bounding Boxes")
    plt.xlabel("Class")
    plt.show()


def check_label_density_per_image(all_labels):
    """
    Check number of bounding boxes per image; plot histogram.
    """
    counts = [len(labels) for labels in all_labels]
    if sum(counts) == 0:
        print("No bounding boxes found in dataset for density check.")
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(counts, bins=30, kde=False)
    plt.title("Number of Bounding Boxes per Image")
    plt.xlabel("Bounding Boxes per Image")
    plt.ylabel("Number of Images")
    plt.show()


def check_unusual_filename_mismatches(image_paths, label_paths):
    """
    Identify label files without corresponding images and vice versa.
    """
    image_stems = {Path(p).stem for p in image_paths}
    label_stems = {Path(p).stem for p in label_paths}

    labels_without_images = label_stems - image_stems
    images_without_labels = image_stems - label_stems

    if labels_without_images:
        print(f"Labels without corresponding images: {len(labels_without_images)}")
    else:
        print("No label files without corresponding images.")

    if images_without_labels:
        print(f"Images without corresponding labels: {len(images_without_labels)}")
    else:
        print("No images without corresponding labels.")

    return list(labels_without_images), list(images_without_labels)


def check_label_file_is_empty_or_corrupt(label_paths):
    """
    Check for empty or unreadable label files.
    """
    empty_files = []
    corrupt_files = []
    for lbl_path in tqdm(label_paths, desc="Checking label files"):
        try:
            if os.path.getsize(lbl_path) == 0:
                empty_files.append(lbl_path)
        except Exception:
            corrupt_files.append(lbl_path)

    if empty_files:
        print(f"Found {len(empty_files)} empty label files (could be used as inference data).")
    else:
        print("No empty label files found.")

    if corrupt_files:
        print(f"Found {len(corrupt_files)} corrupt label files.")
    else:
        print("No corrupt label files found.")

    return empty_files, corrupt_files


def check_label_format_validity(label_paths):
    """
    Check if label files follow YOLO format: 5 parts per line and valid data types.
    """
    invalid_files = []
    for lbl_path in tqdm(label_paths, desc="Checking label format validity"):
        try:
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        invalid_files.append(lbl_path)
                        break
                    class_id = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    if any(c < 0 or c > 1 for c in coords):
                        invalid_files.append(lbl_path)
                        break
        except Exception:
            invalid_files.append(lbl_path)

    if invalid_files:
        print(f"Found {len(invalid_files)} label files with invalid format.")
    else:
        print("All label files conform to YOLO format.")

    return invalid_files


def check_class_id_out_of_yaml_range(all_labels, num_classes):
    """
    Check for labels with class IDs outside valid range from YAML.
    """
    invalid_class_ids = []
    for idx, labels in enumerate(all_labels):
        for label in labels:
            class_id = label[0]
            if class_id < 0 or class_id >= num_classes:
                invalid_class_ids.append((idx, class_id))

    if invalid_class_ids:
        print(f"Found {len(invalid_class_ids)} labels with class IDs outside YAML range.")
    else:
        print("No class ID out of range found.")

    return invalid_class_ids