import os

import cv2


def process_images(input_folder, output_txt, max_size):
    """
    Loop through image files in a folder, display them within a size constraint,
    and append filenames to a text file when the spacebar is pressed.

    Args:
        input_folder (str): Path to the folder containing images.
        output_txt (str): Path to the text file to save image filenames.
        max_size (tuple): Maximum width and height (width, height) for displaying images.
    """
    if not os.path.exists(input_folder):
        print("Input folder does not exist.")
        return

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)

    # Get all image files
    image_files = [
        f
        for f in os.listdir(input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ]

    if not image_files:
        print("No image files found in the input folder.")
        return

    print("Press SPACE to save the image filename, or ESC to quit.")

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipping {image_file}, not a valid image.")
            continue

        # Resize the image to fit within max_size while maintaining aspect ratio
        h, w = image.shape[:2]
        max_width, max_height = max_size
        scale = min(max_width / w, max_height / h, 1)  # Ensure no upscaling
        new_size = (int(w * scale), int(h * scale))
        resized_image = cv2.resize(image, new_size)

        # Display the image
        cv2.imshow("Image Viewer", resized_image)
        key = cv2.waitKey(0)

        # Space key to save filename
        if key == 32:  # Space key
            with open(output_txt, "a") as file:
                file.write(image_file + "\n")
            print(f"Saved: {image_file}")

        # ESC key to exit
        elif key == 27:  # ESC key
            print("Exiting.")
            break

    cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    import sys

    input_folder = sys.argv[1]
    output_txt = sys.argv[2]
    max_size = (800, 600)  # Maximum width and height
    process_images(input_folder, output_txt, max_size)
