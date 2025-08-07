import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import random
import cv2

# Use a consistent style
sns.set(style="whitegrid")

def save_plot(fig, name, save_dir):
    """Save a matplotlib figure to the specified directory."""
    path = save_dir / f"{name}.png"
    fig.savefig(path, bbox_inches='tight')
    print(f"Saved image: {path}")

def plot_class_counts(class_counts, class_names, save_dir):
    """Plot bar chart of class counts."""
    labels = [class_names[cid] if cid < len(class_names) else f"Class {cid}" for cid in class_counts.keys()]
    counts = list(class_counts.values())

    fig, ax = plt.subplots(figsize=(18, 9))
    bars = sns.barplot(x=labels, y=counts, hue=labels, palette="viridis", ax=ax, legend=False)

    ax.set_title("Class Distribution - Bar Chart")
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")

    for bar in bars.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 10, f'{int(height)}', ha='center', fontsize=9)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    save_plot(fig, "class_bar_chart", save_dir)
    plt.show()

def plot_class_distribution_pie(class_counts, class_names, save_dir):
    """Pie chart of class distribution."""
    labels = [class_names[cid] if cid < len(class_names) else f"Class {cid}" for cid in class_counts.keys()]
    sizes = list(class_counts.values())

    fig, ax = plt.subplots(figsize=(18, 18))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    ax.set_title("Class Distribution - Pie Chart")

    save_plot(fig, "class_pie_chart", save_dir)
    plt.show()

def generate_class_distribution_table(class_counts, class_names, save_dir):
    """Create a CSV report of class percentages."""
    total = sum(class_counts.values())
    data = []

    for class_id, count in class_counts.items():
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        percent = round((count / total) * 100, 2)
        data.append([class_id, class_name, count, percent])

    df = pd.DataFrame(data, columns=["Class ID", "Class Name", "Count", "Percentage"])
    save_path = save_dir / "class_distribution.csv"
    df.to_csv(save_path, index=False)
    print(f"📄 Saved CSV report: {save_path}")

def show_random_image_with_boxes(image_paths, all_labels, class_names, save_dir):
    """Display one random image with its bounding boxes."""
    idx = random.randint(0, len(image_paths) - 1)
    img_path = image_paths[idx]
    labels = all_labels[idx]

    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    for label in labels:
        class_id, x, y, bw, bh = label
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"Random Image with Bounding Boxes: {os.path.basename(img_path)}")

    save_plot(fig, "random_image_with_boxes", save_dir)
    plt.show()

def plot_bbox_size_distribution(all_labels, save_dir):
    """Plot distribution of bounding box width & height."""
    widths, heights = [], []

    for labels in all_labels:
        for label in labels:
            _, _, _, w, h = label
            widths.append(w)
            heights.append(h)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(widths, color="skyblue", label="Width", kde=True, ax=ax)
    sns.histplot(heights, color="salmon", label="Height", kde=True, ax=ax)
    ax.set_title("Bounding Box Size Distribution")
    ax.set_xlabel("Normalized Size")
    ax.set_ylabel("Frequency")
    ax.legend()

    save_plot(fig, "bbox_size_distribution", save_dir)
    plt.show()

def plot_image_size_distribution(image_paths, save_dir):
    """Plot distribution of image sizes (in pixels)."""
    widths, heights = [], []

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is not None:
            h, w, _ = img.shape
            heights.append(h)
            widths.append(w)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(widths, label="Width", color="skyblue", kde=True, ax=ax)
    sns.histplot(heights, label="Height", color="salmon", kde=True, ax=ax)
    ax.set_title("Image Size Distribution")
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Frequency")
    ax.legend()

    save_plot(fig, "image_size_distribution", save_dir)
    plt.show()

def plot_annotation_count_per_image(all_labels, save_dir):
    """Plot how many annotations exist per image."""
    ann_counts = [len(labels) for labels in all_labels]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(ann_counts, bins=20, kde=False, color="purple", ax=ax)
    ax.set_title("Annotation Count Per Image")
    ax.set_xlabel("Annotations per Image")
    ax.set_ylabel("Frequency")

    save_plot(fig, "annotation_count_per_image", save_dir)
    plt.show()

def show_class_sample_grid(image_paths, all_labels, class_names, save_dir, samples_per_class=1, grid_cols=4):
    """
    Display a grid of sample images with bounding boxes, one (or more) per class.
    """

    h, w = 160, 160  # Thumbnail size
    class_to_samples = defaultdict(list)

    # Collect one or more samples for each class
    for img_path, labels in zip(image_paths, all_labels):
        for label in labels:
            class_id = int(label[0])
            if len(class_to_samples[class_id]) < samples_per_class:
                class_to_samples[class_id].append((img_path, labels))
            if all(len(v) >= samples_per_class for v in class_to_samples.values()):
                break

    total_classes = len(class_to_samples)
    rows = int(np.ceil(total_classes / grid_cols))
    fig, axes = plt.subplots(rows, grid_cols, figsize=(grid_cols * 4, rows * 4))
    axes = axes.flatten()

    for i, (class_id, samples) in enumerate(class_to_samples.items()):
        img_path, labels = samples[0]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (w, h))

        h_img, w_img, _ = img.shape
        scale_x = w / w_img
        scale_y = h / h_img

        # Draw boxes on the resized image
        for label in labels:
            cid, x, y, bw, bh = label
            if int(cid) != class_id:
                continue
            x1 = int((x - bw / 2) * w_img * scale_x)
            y1 = int((y - bh / 2) * h_img * scale_y)
            x2 = int((x + bw / 2) * w_img * scale_x)
            y2 = int((y + bh / 2) * h_img * scale_y)

            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

        ax = axes[i]
        ax.imshow(img_resized)
        ax.set_title(class_names[class_id] if class_id < len(class_names) else f"Class {class_id}")
        ax.axis('off')

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    save_plot(fig, "class_sample_grid", save_dir)
    plt.show()