import json
import os
from glob import glob
from typing import Dict, List, Tuple

import cv2
from tqdm import tqdm


class YOLOtoCOCO:
    def __init__(self, classes: List[str]):
        self.classes = classes
        self.images = []
        self.annotations = []
        self.categories = []
        self.image_id = 0
        self.annotation_id = 0

        # Create categories
        for class_id, class_name in enumerate(classes):
            category = {"id": class_id, "name": class_name, "supercategory": "none"}
            self.categories.append(category)

    def yolo_to_coco_bbox(
        self, yolo_bbox: List[float], image_width: int, image_height: int
    ) -> List[float]:
        """
        Convert YOLO bbox to COCO bbox format
        YOLO: [x_center, y_center, width, height] (normalized)
        COCO: [x_min, y_min, width, height] (absolute)
        """
        x_center, y_center, width, height = yolo_bbox

        # Denormalize
        x_center = x_center * image_width
        y_center = y_center * image_height
        width = width * image_width
        height = height * image_height

        # Convert to top-left coordinates
        x_min = x_center - (width / 2)
        y_min = y_center - (height / 2)

        return [x_min, y_min, width, height]

    def add_image_and_annotations(self, image_path: str, label_path: str):
        """
        Add image and its annotations to the COCO format
        """
        # Read image to get dimensions
        img = cv2.imread(image_path)
        height, width = img.shape[:2]

        # Add image info
        image = {
            "id": self.image_id,
            "file_name": os.path.basename(image_path),
            "height": height,
            "width": width,
            "license": 1,  # Default license
        }
        self.images.append(image)

        # Read and convert annotations if they exist
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                data = line.strip().split()
                if len(data) == 5:
                    class_id = int(data[0])
                    bbox = [float(x) for x in data[1:]]

                    # Convert YOLO bbox to COCO bbox
                    coco_bbox = self.yolo_to_coco_bbox(bbox, width, height)

                    # Calculate area
                    area = coco_bbox[2] * coco_bbox[3]

                    annotation = {
                        "id": self.annotation_id,
                        "image_id": self.image_id,
                        "category_id": class_id,
                        "bbox": coco_bbox,
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": [],  # Empty segmentation for object detection
                    }
                    self.annotations.append(annotation)
                    self.annotation_id += 1

        self.image_id += 1

    def convert(self, image_dir: str, label_dir: str, output_path: str):
        """
        Convert all YOLO annotations to COCO format
        """
        # Get all image files
        image_files = glob(os.path.join(image_dir, "*.*"))
        image_files = [
            f for f in image_files if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # Process each image and its corresponding label file
        for image_path in tqdm(image_files):
            try:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                label_path = os.path.join(label_dir, f"{base_name}.txt")
                self.add_image_and_annotations(image_path, label_path)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        # Create COCO json structure
        coco_format = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
            "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        }

        # Save to json file
        with open(output_path, "w") as f:
            json.dump(coco_format, f, indent=2)


def main():
    import sys

    # Define your classes (must match the class indices used in YOLO labels)
    classes = ["ship"]  # Update this list with your classes

    # Initialize converter
    converter = YOLOtoCOCO(classes)

    # Define paths
    image_dir = sys.argv[1]  # "images" Directory containing your images
    label_dir = sys.argv[2]  # "labels" Directory containing YOLO label files
    output_path = sys.argv[3]  # "annotations_coco.json" Output COCO JSON file

    # Convert
    converter.convert(image_dir, label_dir, output_path)
    print(f"Conversion completed. COCO format annotations saved to: {output_path}")


if __name__ == "__main__":
    main()
