import os
import yaml
import glob
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# -----------------------------------------------------------
# YAML Handling
# -----------------------------------------------------------

def load_data_yaml(yaml_path):
    """
    Load and parse a YOLO-style data.yaml file.
    """
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    print(f"Loaded YAML config from {yaml_path}: keys={list(data.keys())}")
    return data


def get_class_names(data_yaml):
    """
    Extract class names list from YAML data.
    """
    return data_yaml.get('names', [])


# -----------------------------------------------------------
# Image and Label Path Utilities
# -----------------------------------------------------------

def get_image_paths(split_dir):
    """
    Retrieve all image paths from a given directory using supported extensions.
    """
    image_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_formats:
        found = glob.glob(os.path.join(split_dir, ext))
        if len(found) > 0:
            print(f"Found {len(found)} files for pattern {ext}")
        image_paths.extend(found)
    return sorted(image_paths)


def get_label_paths(image_paths, labels_dir):
    """
    Match each image to its corresponding YOLO label file (.txt).
    """
    label_paths = []
    for img_path in tqdm(image_paths, desc="Generating label paths"):
        filename = Path(img_path).stem
        label_path = os.path.join(labels_dir, f"{filename}.txt")
        label_paths.append(label_path)
    print(f"Generated {len(label_paths)} label paths from {len(image_paths)} images")
    return label_paths


# -----------------------------------------------------------
# Label Loading and Parsing
# -----------------------------------------------------------

def load_labels(label_paths):
    """
    Load label data from a list of label file paths.
    Returns a list of lists: each inner list contains labels for one image.
    """
    all_labels = []
    loaded_count = 0
    failed_files = []

    for label_path in tqdm(label_paths, desc="Loading label files"):
        labels = []
        if not os.path.isfile(label_path):
            all_labels.append(labels)
            continue

        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    class_id, x_center, y_center, width, height = parts
                    labels.append([
                        int(class_id), float(x_center), float(y_center),
                        float(width), float(height)
                    ])
            loaded_count += 1
        except Exception:
            failed_files.append(label_path)
            labels = []

        all_labels.append(labels)

    print(f"Loaded labels from {loaded_count}/{len(label_paths)} files.")
    if failed_files:
        print(f"Failed to load {len(failed_files)} files. Example(s): {failed_files[:3]}")

    return all_labels


def load_all_labels(label_paths):
    """
    Wrapper to load all labels using a single function call.
    """
    return load_labels(label_paths)


# -----------------------------------------------------------
# Dataset Summary
# -----------------------------------------------------------

def summarize_dataset(image_paths, label_paths, all_labels, class_names=None):
    """
    Display a high-level summary of the dataset including class distribution.
    Also reports images without labels and labels without images.
    """
    print("Dataset Summary:")
    total_images = len(image_paths)
    total_labels = len(all_labels)
    empty_labels = sum(1 for labels in all_labels if not labels)

    # Flatten label_paths if needed
    if any(isinstance(lp, list) for lp in label_paths):
        label_paths_flat = [item for sublist in label_paths for item in sublist]
    else:
        label_paths_flat = label_paths

    # Get filenames without extensions to compare images and labels
    image_files = set(os.path.splitext(os.path.basename(p))[0] for p in image_paths)
    label_files = set(os.path.splitext(os.path.basename(p))[0] for p in label_paths_flat)

    # Labels without images
    labels_without_images = label_files - image_files

    print(f"Total images found: {total_images}")
    print(f"Total label files loaded: {total_labels}")
    print(f"Images without labels: {empty_labels}")
    print(f"Labels without images: {len(labels_without_images)}")

    # Class distribution
    class_counts = Counter()
    for labels in all_labels:
        for label in labels:
            class_id = int(label[0])
            class_counts[class_id] += 1

    if class_names:
        print(f"Class distribution ({len(class_names)} classes):")
        for cls_id, count in sorted(class_counts.items()):
            if cls_id < len(class_names):
                print(f"  {class_names[cls_id]} ({cls_id}): {count}")
            else:
                print(f"  Unknown class ID {cls_id}: {count}")
    else:
        print("\nNo class names found in YAML.")

    return class_counts


# -----------------------------------------------------------
# Validation & Checks
# -----------------------------------------------------------

def validate_labels(all_labels, num_classes):
    """
    Validate that all labels contain valid class IDs and normalized bounding boxes.
    Returns:
        - List of invalid class IDs
        - List of labels with coordinates out of range [0, 1]
    """
    invalid_class_ids = []
    invalid_coords = []

    for idx, labels in enumerate(all_labels):
        for label in labels:
            class_id, x, y, w, h = label
            if class_id >= num_classes or class_id < 0:
                invalid_class_ids.append((idx, class_id))
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                invalid_coords.append((idx, label))

    print(f"Found {len(invalid_class_ids)} invalid class ID entries.")
    print(f"Found {len(invalid_coords)} labels with out-of-bound coordinates.")
    return invalid_class_ids, invalid_coords


def check_missing_pairs(image_paths, label_paths):
    """
    Check for missing image or label files in the dataset.
    """
    missing_images = []
    missing_labels = []

    for img_path, lbl_path in zip(image_paths, label_paths):
        if not os.path.exists(img_path):
            missing_images.append(img_path)
        if not os.path.exists(lbl_path):
            missing_labels.append(lbl_path)

    print(f"Missing images: {len(missing_images)}")
    print(f"Missing labels: {len(missing_labels)}")

    return missing_images, missing_labels


# Define function
def check_labels_without_images(label_paths, image_paths):
    from pathlib import Path
    image_stems = {Path(p).stem for p in image_paths}
    labels_without_images = []
    for lbl_path in label_paths:
        stem = Path(lbl_path).stem
        if stem not in image_stems:
            labels_without_images.append(lbl_path)
    print(f"Label files without corresponding images: {len(labels_without_images)}")
    return labels_without_images


import json

def save_dataset_summary_json(
    image_paths,
    label_paths,
    all_labels,
    class_names,
    save_dir
):
    """
    Save a comprehensive dataset summary as JSON including:
    - Dataset size
    - Class distribution
    - QA checks (missing pairs, empty labels, etc.)
    """
    from collections import Counter

    # Prep stats
    total_images = len(image_paths)
    total_labels = len(label_paths)
    total_classes = len(class_names)
    empty_label_images = sum(1 for labels in all_labels if not labels)

    # Count class occurrences
    class_counts = Counter()
    total_annotations = 0
    for labels in all_labels:
        for label in labels:
            class_id = int(label[0])
            class_counts[class_id] += 1
            total_annotations += 1

    # Class stats list
    class_stats = []
    for class_id, count in sorted(class_counts.items()):
        name = class_names[class_id] if class_id < total_classes else f"Class_{class_id}"
        percent = round((count / total_annotations) * 100, 2) if total_annotations else 0.0
        class_stats.append({
            "class_id": class_id,
            "class_name": name,
            "count": count,
            "percentage": percent
        })

    # Check for labels without images
    image_stems = {Path(p).stem for p in image_paths}
    labels_without_images = [
        str(p) for p in label_paths if Path(p).stem not in image_stems
    ]

    # Final summary dictionary
    summary = {
        "dataset_summary": {
            "total_images": total_images,
            "total_label_files": total_labels,
            "total_annotations": total_annotations,
            "images_without_labels": empty_label_images,
            "labels_without_images": len(labels_without_images),
            "total_classes": total_classes
        },
        "class_distribution": class_stats
    }

    # Save JSON
    os.makedirs(save_dir, exist_ok=True)
    output_path = save_dir / "dataset_summary.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Saved dataset summary JSON to {output_path}")


# -----------------------------------------------------------
# Optional: Dataset Object Wrapper
# -----------------------------------------------------------

class YoloDataset:
    """
    A utility class to load and manage YOLO-style datasets.
    """

    def __init__(self, images_dir, labels_dir, yaml_path):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.yaml_path = yaml_path

        # Load and parse YAML
        self.data_yaml = load_data_yaml(yaml_path)
        self.class_names = self.data_yaml.get("names", [])
        self.num_classes = len(self.class_names)

        # Load image and label paths
        self.image_paths = get_image_paths(images_dir)
        self.label_paths = get_label_paths(self.image_paths, labels_dir)
        self.all_labels = load_all_labels(self.label_paths)

        # Identify images without labels
        self.images_without_labels = [
            img for img, labels in zip(self.image_paths, self.all_labels) if not labels
        ]

        # Find label files that do not have corresponding image files
        self.labels_without_images = [
         lbl_path for lbl_path, img_path in zip(self.label_paths, self.image_paths)
            if not os.path.isfile(img_path)
               ]