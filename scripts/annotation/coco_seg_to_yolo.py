import json
import os
from typing import Dict, List


def create_category_mapping(categories: List[Dict]) -> Dict[int, int]:
    """
    Create a mapping of COCO category IDs to YOLO class indices (0-based).
    """
    return {cat["id"]: idx for idx, cat in enumerate(categories)}


def convert_bbox_to_yolo(bbox: List[float], img_width: int, img_height: int) -> tuple:
    """
    Convert COCO bbox [x, y, width, height] to YOLO format [x_center, y_center, width, height]
    All values are normalized between 0 and 1
    """
    x, y, width, height = bbox

    # Calculate normalized center coordinates
    x_center = (x + width / 2) / img_width
    y_center = (y + height / 2) / img_height

    # Calculate normalized width and height
    norm_width = width / img_width
    norm_height = height / img_height

    return x_center, y_center, norm_width, norm_height


def convert_coco_to_yolo(coco_file: str, output_dir: str):
    """
    Convert COCO JSON annotations to YOLO format label files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load COCO JSON file
    with open(coco_file, "r") as f:
        coco_data = json.load(f)

    # Create category ID to YOLO class index mapping
    cat_mapping = create_category_mapping(coco_data["categories"])

    # Create a mapping of image ID to image details
    image_dict = {img["id"]: img for img in coco_data["images"]}

    # Group annotations by image_id
    image_annotations = {}
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)

    # Process each image and create corresponding YOLO label file
    for image_id, annotations in image_annotations.items():
        image_info = image_dict[image_id]
        img_width = image_info["width"]
        img_height = image_info["height"]

        # Create label file name by replacing image extension with .txt
        base_name = os.path.splitext(image_info["file_name"])[0]
        label_file = os.path.join(output_dir, f"{base_name}.txt")

        with open(label_file, "w") as f:
            for ann in annotations:
                # Get YOLO class index
                class_idx = cat_mapping[ann["category_id"]]

                # Convert bbox to YOLO format
                bbox = convert_bbox_to_yolo(ann["bbox"], img_width, img_height)

                # Write line to label file: class_idx x_center y_center width height
                f.write(f"{class_idx} {' '.join([f'{x:.6f}' for x in bbox])}\n")


def main():
    """
    Main function to run the conversion
    """
    # Define input and output paths
    coco_file = "coco_seg_sample.json"  # Update this path to your COCO JSON file
    output_dir = "tmp/yolo_labels"  # Directory where YOLO label files will be saved

    # Run conversion
    convert_coco_to_yolo(coco_file, output_dir)
    print(f"Conversion completed. YOLO labels saved in: {output_dir}")


if __name__ == "__main__":
    main()
