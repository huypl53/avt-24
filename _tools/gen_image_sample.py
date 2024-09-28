import albumentations as A
import cv2
import sys
import os
from PIL import Image


if __name__ == "__main__":

    im_path = sys.argv[1]
    image = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    print(f"Augment {im_path}")

    input_image = Image.open(im_path)
    metadata = input_image.info
    input_image.save(im_path, tiffinfo=metadata)

    base_path, ext = os.path.splitext(im_path)

    exit()
    transform = A.Compose(
        [
            # A.RandomCrop(width=256, height=256),
            # A.HorizontalFlip(p=0.5),
            # A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(
                # always_apply=False,
                p=1.0,
                shift_limit_x=(-0.05, 0.05),
                shift_limit_y=(-0.05, 0.05),
                scale_limit=(-0.05, -0.04),
                rotate_limit=(-15, 15),
                interpolation=2,
                border_mode=2,
                value=(0, 0, 0),
                # mask_value=None,
                rotate_method="largest_box",
            )
        ]
    )
    for i in range(7):
        new_path = f"{base_path}_{i:03d}.{ext}"
        transformed_image = transform(image=image)["image"]
        # cv2.imwrite(
        #     new_path,
        #     transformed_image,
        # )

        output_image = Image.fromarray(
            cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
        )
        output_image.save(new_path, tiffinfo=metadata)
