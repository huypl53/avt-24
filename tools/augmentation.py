import albumentations as A
from albumentations.pytorch import ToTensorV2


class RandomResizedCropRetainSize(A.RandomResizedCrop):
    def __init__(self, scale=(0.9, 1.0), ratio=(0.9, 1.1), p=0.5):
        # Initialize with dummy values that will be updated in apply()
        super().__init__(height=1024, width=1024, scale=scale, ratio=ratio, p=p)

    def apply(self, img, **params):
        height, width = img.shape[:2]
        # Update the height and width before applying the transform
        self.height = height
        self.width = width
        return super().apply(img, **params)


def get_training_augmentation():
    """
    Creates an Albumentations transform pipeline for image segmentation.
    Includes spatial transforms, color augmentations, and noise transforms.

    Args:
        height (int): Target height for resizing
        width (int): Target width for resizing

    Returns:
        A.Compose: Augmentation pipeline that can be applied to both image and mask
    """
    train_transform = A.Compose(
        [
            # Spatial Transforms
            RandomResizedCropRetainSize(
                scale=(0.9, 1.0),  # Crop 90-100% of original size
                ratio=(0.9, 1.1),  # Allow slight aspect ratio changes
                p=0.6,
            ),
            A.Flip(p=0.5),
            A.Rotate(limit=45, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.2, scale_limit=0.2, rotate_limit=30, border_mode=0, p=0.5
            ),
            # Color Transforms (only applied to images, not masks)
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.8
                    ),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.8),
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=0.8,
                    ),
                ],
                p=0.5,
            ),
            # Noise Transforms
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.8),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.8),
                    A.ISONoise(color_shift=(0.01, 0.05), p=0.8),
                ],
                p=0.3,
            ),
            # Normalization and Final Transforms
            # A.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225],
            #     max_pixel_value=255.0,
            #     p=1.0
            # ),
            # ToTensorV2()
        ],
        additional_targets={"mask": "mask"},
    )

    return train_transform


def get_validation_augmentation():
    """
    Creates a minimal transform pipeline for validation/testing.
    Only includes resize and normalization.

    Args:
        height (int): Target height for resizing
        width (int): Target width for resizing

    Returns:
        A.Compose: Basic pipeline for validation/testing
    """
    test_transform = A.Compose(
        [
            # A.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225],
            #     max_pixel_value=255.0,
            #     p=1.0
            # ),
            # ToTensorV2()
        ],
        additional_targets={"mask": "mask"},
    )

    return test_transform
