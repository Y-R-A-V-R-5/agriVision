import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import random
import cv2
from tqdm import tqdm
import random

# Use a consistent style
sns.set(style="whitegrid")


# --------------------------
# SECTION 0: General Tools
# --------------------------

def save_plot(fig, name, save_dir):
    """Save a matplotlib figure to the specified directory."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{name}.png"
    fig.savefig(path, bbox_inches='tight')
    print(f"Saved image: {path}")


def show_classwise_images_with_and_without_boxes(
    image_paths,
    all_labels,
    class_names,
    save_dir,
    samples_per_class=3,
    grid_cols=3,
    use_random=True
):
    """
    Show sample images per class (grouped class-wise), with and without bounding boxes.

    Each class will have:
        - Top row: with bounding boxes
        - Bottom row: without bounding boxes
    """

    def draw_boxes(img, labels, class_id, color=(0, 255, 0)):
        h_img, w_img = img.shape[:2]
        for label in labels:
            cid, x, y, bw, bh = label
            if int(cid) != class_id:
                continue
            x1 = int((x - bw / 2) * w_img)
            y1 = int((y - bh / 2) * h_img)
            x2 = int((x + bw / 2) * w_img)
            y2 = int((y + bh / 2) * h_img)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        return img

    class_to_samples = defaultdict(list)

    for img_path, labels in zip(image_paths, all_labels):
        for label in labels:
            cid = int(label[0])
            if len(class_to_samples[cid]) < samples_per_class * 2:
                class_to_samples[cid].append((img_path, labels))

    for class_id, samples in class_to_samples.items():
        selected = random.sample(samples, min(samples_per_class, len(samples))) if use_random else samples[:samples_per_class]

        fig, axs = plt.subplots(2, grid_cols, figsize=(grid_cols * 4, 8))
        axs = axs.flatten()

        for i in range(grid_cols):
            if i < len(selected):
                img_path, labels = selected[i]
                img = cv2.imread(str(img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_with = draw_boxes(img_rgb.copy(), labels, class_id)
                img_resized_with = cv2.resize(img_with, (160, 160))
                img_resized_plain = cv2.resize(img_rgb, (160, 160))

                axs[i].imshow(img_resized_with)
                axs[i].set_title("With Boxes")
                axs[i].axis('off')

                axs[i + grid_cols].imshow(img_resized_plain)
                axs[i + grid_cols].set_title("Without Boxes")
                axs[i + grid_cols].axis('off')
            else:
                axs[i].axis('off')
                axs[i + grid_cols].axis('off')

        fig.suptitle(f"Class: {class_names[class_id]}", fontsize=14)
        plt.tight_layout()
        filename = f"class_{class_id}_{class_names[class_id].replace(' ', '_')}_samples"
        save_plot(fig, filename, save_dir)
        plt.show()

  
def show_random_images_with_and_without_boxes(
    image_paths,
    all_labels,
    class_names,
    save_dir,
    sample_size=3
):
    """
    Show N random images (from any class) with and without bounding boxes.
    """
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    indices = random.sample(range(len(image_paths)), min(sample_size, len(image_paths)))
    fig, axs = plt.subplots(2, sample_size, figsize=(sample_size * 4, 8))

    for i, idx in enumerate(indices):
        img_path = image_paths[idx]
        labels = all_labels[idx]

        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_with = img_rgb.copy()

        h_img, w_img = img.shape[:2]
        for label in labels:
            cid, x, y, bw, bh = label
            x1 = int((x - bw / 2) * w_img)
            y1 = int((y - bh / 2) * h_img)
            x2 = int((x + bw / 2) * w_img)
            y2 = int((y + bh / 2) * h_img)
            color = tuple(int(x) for x in np.random.randint(0, 255, 3))
            class_name = class_names[int(cid)] if int(cid) < len(class_names) else f"Class {cid}"
            cv2.rectangle(img_with, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_with, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        axs[0, i].imshow(cv2.resize(img_with, (160, 160)))
        axs[0, i].axis('off')
        axs[0, i].set_title("With Boxes")

        axs[1, i].imshow(cv2.resize(img_rgb, (160, 160)))
        axs[1, i].axis('off')
        axs[1, i].set_title("Without Boxes")

    plt.tight_layout()
    save_plot(fig, "random_images_with_and_without_boxes", save_dir)
    plt.show()


# --------------------------
# SECTION 1: Class Analysis
# --------------------------

def plot_class_distribution_combined(class_counts, class_names, save_dir, top_n=None):
    """
    Plot class distribution using both bar and pie charts.
    If top_n is specified, shows only the top N classes by count.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Sort by count (descending)
    sorted_counts = dict(sorted(class_counts.items(), key=lambda item: item[1], reverse=True))

    # Optionally filter to top N
    if top_n is not None:
        sorted_counts = dict(list(sorted_counts.items())[:top_n])
        title_suffix = f"Top {top_n} Classes"
        filename_suffix = f"_top_{top_n}"
    else:
        title_suffix = "All Classes"
        filename_suffix = ""

    total = sum(class_counts.values())  # still total of all, not just top_n
    labels = [class_names[cid] if cid < len(class_names) else f"Class {cid}" for cid in sorted_counts.keys()]
    counts = list(sorted_counts.values())
    percentages = [count / total * 100 for count in counts]

    # Create combined subplot
    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(26, 11))

    # -------- Bar Chart --------
    bars = sns.barplot(x=labels, y=counts, palette="magma", ax=ax_bar)
    ax_bar.set_title(f"{title_suffix} - Bar Chart")
    ax_bar.set_ylabel("Count")
    ax_bar.set_xlabel("Class")
    ax_bar.set_xticklabels(labels, rotation=45, ha='right')

    for bar, pct in zip(bars.patches, percentages):
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, height + 5,
                    f'{int(height)}\n({pct:.1f}%)', ha='center', fontsize=9)

    # -------- Pie Chart --------
    ax_pie.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140,
               colors=sns.color_palette("magma", len(counts)))
    ax_pie.axis('equal')
    ax_pie.set_title(f"{title_suffix} - Pie Chart")

    plt.tight_layout()
    save_plot(fig, f"class_distribution_combined{filename_suffix}", save_dir)
    plt.show()


def plot_class_cooccurrence_heatmap(all_labels, class_names, save_dir):
    """Plot a heatmap showing class co-occurrence across images."""
    num_classes = len(class_names)
    cooccurrence = np.zeros((num_classes, num_classes), dtype=int)

    for labels in all_labels:
        classes_in_image = set([int(label[0]) for label in labels])
        for c1 in classes_in_image:
            for c2 in classes_in_image:
                if c1 != c2:
                    cooccurrence[c1, c2] += 1

    df_cooc = pd.DataFrame(cooccurrence, index=class_names, columns=class_names)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df_cooc, annot=True, fmt='d', cmap='coolwarm', ax=ax)
    ax.set_title("Class Co-occurrence Heatmap")

    plt.tight_layout()
    save_plot(fig, "class_cooccurrence_heatmap", save_dir)
    plt.show()


# --------------------------
# SECTION 2: Image Quality
# --------------------------

def plot_rgb_distribution(image_paths, save_dir):
    """
    Plot average RGB channel intensities across the dataset as a line graph.
    Each channel (R, G, B) will have its average pixel intensity per image plotted.
    """
    avg_r, avg_g, avg_b = [], [], []

    for path in tqdm(image_paths, desc="Computing RGB intensities", leave=False):
        img = cv2.imread(str(path))
        if img is not None:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            avg_r.append(np.mean(img_rgb[:, :, 0]))
            avg_g.append(np.mean(img_rgb[:, :, 1]))
            avg_b.append(np.mean(img_rgb[:, :, 2]))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(avg_r, label='Red', color='red')
    ax.plot(avg_g, label='Green', color='green')
    ax.plot(avg_b, label='Blue', color='blue')

    ax.set_title("Average RGB Intensity per Image")
    ax.set_xlabel("Image Index")
    ax.set_ylabel("Average Pixel Intensity")
    ax.legend()
    plt.tight_layout()

    save_plot(fig, "rgb_intensity_distribution", save_dir)
    plt.show()

def plot_rgb_distribution_per_class(image_paths, all_labels, class_names, save_dir):
    """
    Plot average RGB channel intensities per class.

    Args:
        image_paths (list): List of image paths.
        all_labels (list): List of labels per image. Each element is a list of (class_id, x, y, w, h).
        class_names (list): List of class names indexed by class_id.
        save_dir (str): Directory to save the plot.
    """
    class_rgb = defaultdict(lambda: {'r': [], 'g': [], 'b': []})

    print("[*] Computing RGB intensities per class...")
    for img_path, labels in tqdm(zip(image_paths, all_labels), total=len(image_paths), desc="Processing images", leave=False):
        if not labels:
            continue  # Skip images without labels

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        avg_r = np.mean(img_rgb[:, :, 0])
        avg_g = np.mean(img_rgb[:, :, 1])
        avg_b = np.mean(img_rgb[:, :, 2])

        unique_class_ids = set(int(label[0]) for label in labels)
        for class_id in unique_class_ids:
            class_rgb[class_id]['r'].append(avg_r)
            class_rgb[class_id]['g'].append(avg_g)
            class_rgb[class_id]['b'].append(avg_b)

    # Aggregate RGB values per class
    class_ids = sorted(class_rgb.keys())
    avg_r = [np.mean(class_rgb[cid]['r']) for cid in class_ids]
    avg_g = [np.mean(class_rgb[cid]['g']) for cid in class_ids]
    avg_b = [np.mean(class_rgb[cid]['b']) for cid in class_ids]
    class_labels = [class_names[cid] if cid < len(class_names) else f"Class {cid}" for cid in class_ids]

    # Plot
    x = np.arange(len(class_ids))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width, avg_r, width, label='Red', color='red')
    ax.bar(x, avg_g, width, label='Green', color='green')
    ax.bar(x + width, avg_b, width, label='Blue', color='blue')

    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=45, ha='right')
    ax.set_ylabel("Average RGB Intensity")
    ax.set_title("Average RGB Intensity per Class")
    ax.legend()

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "rgb_intensity_per_class.png")
    plt.savefig(path)
    print(f"Saved RGB class-wise plot to: {path}")
    plt.show()

def plot_image_quality_metrics_all(image_paths, save_dir):
    """
    Compute and plot image quality metrics for all images:
    blur, contrast, brightness, saturation, sharpness, darkness.
    Visualizes using box plots for better comparison.
    """

    def measure_blur(img):
        return cv2.Laplacian(img, cv2.CV_64F).var()

    def measure_contrast(img):
        return img.std()

    def measure_brightness(img):
        return img.mean()

    def measure_saturation(img_rgb):
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        return hsv[:, :, 1].mean()

    def measure_sharpness(img):
        return cv2.Laplacian(img, cv2.CV_64F).var()

    def measure_darkness(img_rgb):
        return np.percentile(img_rgb, 10)

    quality_data = {
        "filename": [],
        "blur": [],
        "contrast": [],
        "brightness": [],
        "saturation": [],
        "sharpness": [],
        "darkness": []
    }

    print("[*] Measuring image quality metrics for all images...")
    for path in tqdm(image_paths, desc="Processing images", leave=False):
        img = cv2.imread(str(path))
        if img is None:
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        quality_data["filename"].append(os.path.basename(path))
        quality_data["blur"].append(measure_blur(img_gray))
        quality_data["contrast"].append(measure_contrast(img_gray))
        quality_data["brightness"].append(measure_brightness(img_rgb))
        quality_data["saturation"].append(measure_saturation(img_rgb))
        quality_data["sharpness"].append(measure_sharpness(img_gray))
        quality_data["darkness"].append(measure_darkness(img_rgb))

    df = pd.DataFrame(quality_data)

    # Box plots for each metric
    melted = df.melt(id_vars="filename", var_name="Metric", value_name="Value")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=melted, x="Metric", y="Value", ax=ax, palette="Set2")
    ax.set_title("Image Quality Metrics Across All Images")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(save_dir, "image_quality_metrics_all.png")
    plt.savefig(plot_path)
    print(f"Saved image quality plot to: {plot_path}")
    plt.show()

def plot_image_quality_by_class(image_paths, all_labels, class_names, save_dir):
    def measure_blur(img): return cv2.Laplacian(img, cv2.CV_64F).var()
    def measure_contrast(img): return img.std()
    def measure_brightness(img): return img.mean()
    def measure_saturation(img_rgb): return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].mean()
    def measure_sharpness(img): return cv2.Laplacian(img, cv2.CV_64F).var()
    def measure_darkness(img_rgb): return np.percentile(img_rgb, 10)

    metrics = {
        'Blur': measure_blur,
        'Contrast': measure_contrast,
        'Brightness': measure_brightness,
        'Saturation': measure_saturation,
        'Sharpness': measure_sharpness,
        'Darkness': measure_darkness,
    }

    # Collect metrics per class
    metric_data = defaultdict(lambda: defaultdict(list))  # metric_data[metric][class] = [values]

    for path, labels in tqdm(zip(image_paths, all_labels), total=len(image_paths), desc="Computing metrics", leave=False):
        if not labels:
            continue
        # Use the first class or dominant class
        class_id = int(labels[0][0])
        if class_id >= len(class_names):
            continue

        img = cv2.imread(str(path))
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate and store each metric
        for metric_name, func in metrics.items():
            if metric_name in ['Blur', 'Contrast', 'Sharpness']:
                value = func(img_gray)
            else:
                value = func(img_rgb)
            metric_data[metric_name][class_names[class_id]].append(value)

    # Plot each metric as a boxplot per class
    for metric_name, class_values in metric_data.items():
        df = []
        for class_name, values in class_values.items():
            for v in values:
                df.append({'Class': class_name, metric_name: v})
        df = pd.DataFrame(df)

        plt.figure(figsize=(16, 8))
        sns.boxplot(data=df, x='Class', y=metric_name)
        plt.xticks(rotation=45, ha='right')
        plt.title(f"{metric_name} Distribution per Class")
        plt.tight_layout()

        filename = f"{metric_name.lower()}_by_class"
        save_plot(plt.gcf(), filename, save_dir)
        plt.show()


def plot_histogram_variance(image_paths, save_dir):
    """
    Plot histogram of pixel intensity variance across images.
    Optionally save variance data for external analysis.

    Args:
        image_paths (list): List of image paths.
        save_dir (str): Directory to save the plot and optional data.
        save_data (bool): Whether to save variance values.
        output_format (str): 'csv' or 'json'
    """

    variances = []
    filenames = []

    print("[*] Calculating image variances...")
    for path in tqdm(image_paths, desc="Processing images", leave=False):
        try:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                var = np.var(img)
                variances.append(var)
                filenames.append(os.path.basename(path))
        except Exception as e:
            print(f"Warning: Could not process {path} — {e}")

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(variances, bins=30, kde=True, color='purple', ax=ax)
    ax.set_title("Histogram of Pixel Intensity Variance")
    ax.set_xlabel("Variance")
    ax.set_ylabel("Frequency")
    plt.tight_layout()

    plot_path = os.path.join(save_dir, "pixel_intensity_variance_histogram.png")
    fig.savefig(plot_path)
    print(f"Saved plot to: {plot_path}")
    plt.show()


def plot_histogram_variance_by_class(image_paths, all_labels, class_names, save_dir):
    """
    Plot histogram of pixel intensity variance across images (overall + per class).
    Assumes image_paths and all_labels are aligned 1:1.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Store variances for global and per-class plots
    overall_variances = []
    class_variance_map = defaultdict(list)

    print("[*] Calculating variances...")
    for img_path, labels in tqdm(zip(image_paths, all_labels), total=len(image_paths), desc="Processing images", leave=False):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        var = np.var(img)
        overall_variances.append(var)

        # Collect class-wise variances (may be multiple per image)
        class_ids = list(set([int(lbl[0]) for lbl in labels]))
        for cid in class_ids:
            class_variance_map[cid].append(var)

    # --- Per-Class Histograms ---
    for cid, variances in class_variance_map.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(variances, bins=30, kde=True, color='teal', ax=ax)
        class_name = class_names[cid] if cid < len(class_names) else f"Class {cid}"
        ax.set_title(f"Pixel Variance Histogram - {class_name}")
        ax.set_xlabel("Variance")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        fname = f"variance_histogram_class_{cid}_{class_name.replace(' ', '_')}.png"
        fig.savefig(os.path.join(save_dir, fname))
        print(f"Saved: {fname}")
        plt.show()

# --------------------------
# SECTION 3: Bounding Boxes
# --------------------------

def plot_bbox_size_distribution(all_labels, save_dir):
    """
    Plot distribution of bounding box widths and heights (normalized).
    """
    widths, heights = [], []

    for labels in all_labels:
        for label in labels:
            _, _, _, w, h = label
            widths.append(w)
            heights.append(h)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(widths, color="skyblue", label="Width", kde=True, ax=ax)
    sns.histplot(heights, color="salmon", label="Height", kde=True, ax=ax)
    ax.set_title("Bounding Box Size Distribution (Normalized)")
    ax.set_xlabel("Normalized Size")
    ax.set_ylabel("Frequency")
    ax.legend()

    save_plot(fig, "bbox_size_distribution", save_dir)
    plt.show()


def plot_bbox_area_distribution(all_labels, save_dir):
    """
    Plot distribution of bounding box areas (width * height, normalized).
    """
    areas = []

    for labels in all_labels:
        for label in labels:
            _, _, _, w, h = label
            areas.append(w * h)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(areas, bins=50, color="teal", kde=True, ax=ax)
    ax.set_title("Bounding Box Area Distribution (Normalized)")
    ax.set_xlabel("Area (Width x Height)")
    ax.set_ylabel("Frequency")

    save_plot(fig, "bbox_area_distribution", save_dir)
    plt.show()


def plot_annotation_count_per_image(all_labels, save_dir):
    """
    Plot histogram of annotation counts (number of bounding boxes) per image.
    """
    ann_counts = [len(labels) for labels in all_labels]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(ann_counts, bins=20, kde=False, color="purple", ax=ax)
    ax.set_title("Annotation Count Per Image")
    ax.set_xlabel("Annotations per Image")
    ax.set_ylabel("Frequency")

    save_plot(fig, "annotation_count_per_image", save_dir)
    plt.show()


def plot_bbox_density_heatmap(image_paths, all_labels, grid_size=(10, 10), save_dir=None):
    """
    Plot a heatmap showing spatial density of bounding boxes over the dataset.
    Divide images into a grid and count how many bounding boxes fall in each cell.
    Assumes all images have the same size or normalized coordinates.
    """
    heatmap = np.zeros(grid_size)

    for labels in all_labels:
        for label in labels:
            _, x_center, y_center, _, _ = label
            # x_center and y_center are normalized (0-1)
            x_idx = min(int(x_center * grid_size[1]), grid_size[1]-1)
            y_idx = min(int(y_center * grid_size[0]), grid_size[0]-1)
            heatmap[y_idx, x_idx] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heatmap, cmap="YlOrRd", ax=ax)
    ax.set_title("Bounding Box Spatial Density Heatmap")
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")

    if save_dir:
        save_plot(fig, "bbox_density_heatmap", save_dir)
    plt.show()

# --------------------------
# SECTION 4: Outlier Detection
# --------------------------

def detect_low_variance_images(image_paths, threshold=500):
    """
    Identify images with low variance (likely very uniform/dark/blank images).
    Returns a list of tuples (image_path, variance).
    """
    low_variance_images = []

    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        variance = np.var(img)
        if variance < threshold:
            low_variance_images.append((path, variance))

    return low_variance_images


def flag_extreme_bbox_sizes(all_labels, area_threshold_low=0.001, area_threshold_high=0.3):
    """
    Identify bounding boxes with extreme small or large normalized areas.
    Returns dict with 'too_small' and 'too_large' keys containing lists of (image_idx, bbox).
    """
    too_small = []
    too_large = []

    for img_idx, labels in enumerate(all_labels):
        for bbox in labels:
            _, _, _, w, h = bbox
            area = w * h
            if area < area_threshold_low:
                too_small.append((img_idx, bbox))
            elif area > area_threshold_high:
                too_large.append((img_idx, bbox))

    return {"too_small": too_small, "too_large": too_large}


def show_outlier_images(image_paths, outlier_indices, max_display=5, save_dir=None):
    """
    Display images flagged as outliers (given by indices).
    """
    n_show = min(len(outlier_indices), max_display)
    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 5))
    if n_show == 1:
        axes = [axes]

    for i, idx in enumerate(outlier_indices[:n_show]):
        img = cv2.imread(str(image_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Outlier Image {idx}")

    plt.tight_layout()
    if save_dir:
        save_plot(plt.gcf(), "outlier_images", save_dir)
    plt.show()