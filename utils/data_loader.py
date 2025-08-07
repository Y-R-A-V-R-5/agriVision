import os
import yaml
import glob
from pathlib import Path
from tqdm import tqdm
from collections import Counter

def load_data_yaml(yaml_path):
    """Load and parse a YOLO-style data.yaml file."""
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    print(f"Loaded YAML config from {yaml_path}: keys={list(data.keys())}")
    return data

def get_class_names(data_yaml):
    """Extract class names list from YAML data."""
    return data_yaml.get('names', [])

def get_image_paths(split_dir):
    image_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_formats:
        found = glob.glob(os.path.join(split_dir, ext))
        if len(found) > 0:
            print(f"Found {len(found)} files for pattern {ext}")
        image_paths.extend(found)
    return sorted(image_paths)

def get_label_paths(image_paths, labels_dir):
    """Match each image to its corresponding label file."""
    label_paths = []
    for img_path in tqdm(image_paths, desc="Generating label paths"):
        filename = Path(img_path).stem
        label_path = os.path.join(labels_dir, f"{filename}.txt")
        label_paths.append(label_path)
    print(f"Generated {len(label_paths)} label paths from {len(image_paths)} images")
    return label_paths

from tqdm import tqdm
import os

def load_labels(label_paths):
    """
    Load label data from a list of label file paths.
    Returns a list of lists: each inner list contains labels for one image.
    """
    all_labels = []
    loaded_count = 0
    failed_files = []

    for label_path in tqdm(label_paths, desc="Loading label files"):
        labels = []  # <- always define labels here
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
        except Exception as e:
            failed_files.append(label_path)
            labels = []  # If error, make sure to define this fallback

        all_labels.append(labels)

    print(f"Loaded labels from {loaded_count}/{len(label_paths)} files.")
    if failed_files:
        print(f"Failed to load {len(failed_files)} files. Example(s): {failed_files[:3]}")

    return all_labels

def load_all_labels(label_paths):
    """Wrapper that calls load_labels on the full list."""
    return load_labels(label_paths)

def summarize_dataset(image_paths, all_labels, class_names=None):
    """Display a high-level summary of the dataset."""
    print("\nDataset Summary:")
    total_images = len(image_paths)
    total_labels = len(all_labels)
    empty_labels = sum(1 for labels in all_labels if not labels)

    print(f"Total images found: {total_images}")
    print(f"Total label files loaded: {total_labels}")
    print(f"Images without labels: {empty_labels}")

    # Class distribution
    class_counts = Counter()
    for labels in all_labels:
        for label in labels:
            class_id = int(label[0])
            class_counts[class_id] += 1

    if class_names:
        print(f"\nClass distribution ({len(class_names)} classes):")
        for cls_id, count in sorted(class_counts.items()):
            if cls_id < len(class_names):
                print(f"  {class_names[cls_id]} ({cls_id}): {count}")
            else:
                print(f"  Unknown class ID {cls_id}: {count}")
    else:
        print("\nNo class names found in YAML.")

class YoloDataset:
    def __init__(self, images_dir, labels_dir, yaml_path):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.yaml_path = yaml_path

        # Load and parse YAML
        self.data_yaml = load_data_yaml(yaml_path)
        self.class_names = self.data_yaml.get("names", [])
        self.num_classes = len(self.class_names)

        # Load images and labels
        self.image_paths = get_image_paths(images_dir)
        self.label_paths = get_label_paths(self.image_paths, labels_dir)
        self.all_labels = load_all_labels(self.label_paths)  # ← THIS LINE is required

        # Extra: images with no labels
        self.images_without_labels = [
            img for img, labels in zip(self.image_paths, self.all_labels) if not labels
        ]