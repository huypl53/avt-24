import json
import os
from pathlib import Path


def clean_coco_annotations(annotation_path, image_dir):
    """
    Clean COCO annotation file by removing entries for missing images.

    Args:
        annotation_path (str): Path to the COCO annotation JSON file
        image_dir (str): Directory containing the images

    Returns:
        dict: Cleaned annotation dictionary
    """
    # Load the annotation file
    with open(annotation_path, "r") as f:
        coco_data = json.load(f)

    # Get list of existing image files
    image_files = set(os.path.basename(str(p)) for p in Path(image_dir).glob("*"))

    # Keep track of valid image IDs
    valid_image_ids = set()

    # Filter images list
    filtered_images = []
    for img in coco_data["images"]:
        if img["file_name"] in image_files:
            filtered_images.append(img)
            valid_image_ids.add(img["id"])

    # Filter annotations list
    filtered_annotations = [
        ann for ann in coco_data["annotations"] if ann["image_id"] in valid_image_ids
    ]

    # Update the COCO data
    coco_data["images"] = filtered_images
    coco_data["annotations"] = filtered_annotations

    # Save the cleaned annotations
    output_path = annotation_path.replace(".json", "_cleaned.json")
    with open(output_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    # Print statistics
    print(f"Original number of images: {len(coco_data['images'])}")
    print(f"Original number of annotations: {len(coco_data['annotations'])}")
    print(
        f"Number of missing images: {len(coco_data['images']) - len(filtered_images)}"
    )
    print(f"Cleaned annotations saved to: {output_path}")

    return coco_data


# Example usage:
if __name__ == "__main__":
    import sys

    annotation_path = sys.argv[1]  # annotation.json
    image_dir = sys.argv[2]  # images/
    clean_coco_annotations(annotation_path, image_dir)
