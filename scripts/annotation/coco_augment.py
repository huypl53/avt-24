import json
import os
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm


def create_ship_detection_augmentation_pipeline(p=1.0):
    """
    Create a specialized augmentation pipeline for SAR ship detection.

    :param p: Probability of applying the entire augmentation pipeline
    :return: Albumentations Compose object with ship detection-specific augmentations
    """
    return A.Compose(
        [
            # Geometric Transformations
            A.ShiftScaleRotate(
                shift_limit=0.1,  # Slight positional shifts
                scale_limit=0.15,  # Moderate scaling
                rotate_limit=30,  # Rotation up to 30 degrees
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,  # Black border for SAR images
                p=0.7,
            ),
            # Spatial-Aware Augmentations
            A.OneOf(
                [
                    A.Perspective(
                        scale=(0.05, 0.1),  # Moderate perspective distortion
                        pad_mode=cv2.BORDER_CONSTANT,
                        p=1.0,
                    ),
                    A.ElasticTransform(
                        alpha=20,
                        sigma=5,
                        interpolation=cv2.INTER_LINEAR,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=0.3,
                    ),
                ],
                p=0.4,
            ),
            # Intensity Transformations
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        brightness_by_max=False,
                        p=0.5,
                    ),
                    A.CLAHE(
                        clip_limit=2.0,
                        tile_grid_size=(8, 8),
                        p=0.5,
                    ),
                ],
                p=0.6,
            ),
            # Noise Augmentations (Simulating SAR Speckle Noise)
            A.OneOf(
                [
                    # Reduced variance for Gaussian noise
                    A.GaussNoise(
                        var_limit=(5, 25),  # Reduced from (10, 50)
                        mean=0,
                        p=0.5,
                    ),
                    # Reduced intensity for speckle noise
                    A.Lambda(
                        name="speckle_noise",
                        image=lambda img, **params: np.clip(
                            img
                            * np.random.normal(1, 0.05, img.shape),  # Reduced from 0.1
                            0,
                            255,
                        ).astype(np.uint8),
                    ),
                ],
                p=0.3,  # Reduced from 0.4
            ),
            # Optional: Edge Enhancement
            # A.OneOf(
            #     [
            #         A.Lambda(
            #             name="unsharp_mask",
            #             image=lambda img, **params: cv2.addWeighted(
            #                 img, 1.5, cv2.GaussianBlur(img, (0, 0), 3), -0.5, 0
            #             ),
            #         ),
            #         A.Lambda(
            #             name="sobel_edges",
            #             image=lambda img, **params: cv2.Sobel(
            #                 img, cv2.CV_64F, 1, 1, ksize=3
            #             ).astype(np.uint8),
            #         ),
            #     ],
            #     p=0.3,
            # ),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["labels"],
            min_area=100,  # Minimum area to keep a bounding box
            min_visibility=0.3,  # Minimum visibility ratio to keep a bounding box
        ),
        p=p,
    )


def filter_valid_bboxes(
    bboxes: List[List[float]], image_shape: Tuple[int, int]
) -> List[List[float]]:
    """
    Filter out invalid bounding boxes.

    :param bboxes: List of bounding boxes in COCO format [x, y, width, height]
    :param image_shape: Shape of the image (height, width)
    :return: Filtered list of valid bounding boxes
    """
    valid_bboxes = []
    height, width = image_shape

    for bbox in bboxes:
        x, y, w, h = bbox

        # Check if bbox is within image boundaries
        if x >= 0 and y >= 0 and x + w <= width and y + h <= height and w > 0 and h > 0:
            valid_bboxes.append(bbox)

    return valid_bboxes


def augment_image_with_ship_annotations(
    image_path: str, annotations: List[Dict], transform: A.Compose
) -> Tuple[np.ndarray, List[List[float]], List[int]]:
    """
    Augment a SAR image while preserving ship annotations.

    :param image_path: Path to the input image
    :param annotations: List of COCO annotations for the image
    :param transform: Albumentations transformation pipeline
    :return: Augmented image, transformed bounding boxes, and labels
    """
    # Read image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Normalize image
    image = cv2.normalize(
        image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # Convert to 3-channel for Albumentations (required for some transforms)
    image_3channel = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Prepare bounding boxes and labels
    bboxes = []
    labels = []
    for ann in annotations:
        bboxes.append(ann["bbox"])
        labels.append(ann["category_id"])

    # Filter initial bounding boxes
    bboxes = filter_valid_bboxes(bboxes, image.shape)

    # Apply augmentation
    try:
        transformed = transform(image=image_3channel, bboxes=bboxes, labels=labels)
    except Exception as e:
        print(f"Augmentation failed: {e}")
        return image, bboxes, labels

    # Convert back to grayscale
    augmented_image = cv2.cvtColor(transformed["image"], cv2.COLOR_RGB2GRAY)

    # Post-process bounding boxes
    augmented_bboxes = []
    augmented_labels = []
    for bbox, label in zip(transformed["bboxes"], transformed["labels"]):
        # Additional filtering for augmented bboxes
        x, y, w, h = bbox
        if (
            w > 10
            and h > 10  # Minimum size
            and 0 <= x < augmented_image.shape[1]
            and 0 <= y < augmented_image.shape[0]
        ):
            augmented_bboxes.append(bbox)
            augmented_labels.append(label)

    return augmented_image, augmented_bboxes, augmented_labels


def augment_ship_detection_dataset(
    images_txt_path: str,
    images_dir: str,
    coco_json_path: str,
    output_dir: str,
    num_augmentations: int = 3,
):
    """
    Augment SAR ship detection images from a text file using COCO annotations.

    :param images_txt_path: Path to text file containing image filenames
    :param images_dir: Directory containing the actual images
    :param coco_json_path: Path to COCO annotation file
    :param output_dir: Directory to save augmented images
    :param num_augmentations: Number of augmentations per image
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load image list and COCO annotations
    with open(images_txt_path, "r") as f:
        image_filenames = [line.strip() for line in f]

    # Load COCO annotations
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    # Create annotation lookup
    annotations_lookup = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_lookup:
            annotations_lookup[img_id] = []
        annotations_lookup[img_id].append(ann)

    # Create SAR-specific augmentation pipeline
    transform = create_ship_detection_augmentation_pipeline()

    # Initialize lists for new images and annotations
    new_images = []
    new_annotations = []
    next_image_id = max(img["id"] for img in coco_data["images"]) + 1
    next_ann_id = max(ann["id"] for ann in coco_data["annotations"]) + 1

    # Augment images
    for img_filename in tqdm(image_filenames):
        # Find corresponding image in COCO data
        img_info = next(
            (img for img in coco_data["images"] if img["file_name"] == img_filename),
            None,
        )

        if not img_info:
            print(f"Skipping {img_filename}: No COCO annotation found")
            continue

        # Get annotations for this image
        annotations = annotations_lookup.get(img_info["id"], [])

        # Get full path to image using images_dir instead
        img_path = os.path.join(images_dir, img_filename)

        for i in range(num_augmentations):
            # Augment image
            aug_image, aug_bboxes, aug_labels = augment_image_with_ship_annotations(
                img_path, annotations, transform
            )

            # Generate output filename
            base_name = os.path.splitext(img_filename)[0]
            output_filename = f"{base_name}_aug_{i+1}.png"
            output_path = os.path.join(output_dir, output_filename)

            # Save augmented image
            cv2.imwrite(output_path, aug_image)

            # Create new image entry
            new_image = {
                "id": next_image_id,
                "file_name": output_filename,
                "width": aug_image.shape[1],
                "height": aug_image.shape[0],
            }
            new_images.append(new_image)

            # Create new annotations
            for bbox, label in zip(aug_bboxes, aug_labels):
                ann = {
                    "id": next_ann_id,
                    "image_id": next_image_id,
                    "category_id": label,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
                new_annotations.append(ann)
                next_ann_id += 1

            next_image_id += 1

            print(f"Augmented {img_filename} -> {output_filename}")

    # Merge original and new annotations
    merged_coco = {
        "images": coco_data["images"] + new_images,
        "annotations": coco_data["annotations"] + new_annotations,
        "categories": coco_data["categories"],
    }

    # Save merged COCO annotations
    output_json_path = os.path.join(output_dir, "merged_annotations.json")
    with open(output_json_path, "w") as f:
        json.dump(merged_coco, f)

    print(f"Saved merged annotations to {output_json_path}")


# Example usage
if __name__ == "__main__":
    import sys

    IMAGES_TXT_PATH = sys.argv[1]  # Path to the text file with image filenames
    IMAGES_DIR = sys.argv[2]  # Directory containing the images
    COCO_JSON_PATH = sys.argv[3]  # Path to the COCO JSON annotation file
    OUTPUT_DIR = sys.argv[4]  # Directory to save augmented images

    augment_ship_detection_dataset(
        images_txt_path=IMAGES_TXT_PATH,
        images_dir=IMAGES_DIR,
        coco_json_path=COCO_JSON_PATH,
        output_dir=OUTPUT_DIR,
    )
