import cv2
import numpy as np
from sklearn.cluster import DBSCAN


def preprocess_image(image):
    """Preprocess the image with focus on runway features"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply bilateral filter to preserve edges while removing noise
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

    return denoised


def create_runway_mask(image):
    """Create a mask to focus on potential runway areas"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Detect grayish/concrete colors typical for runways
    lower_gray = np.array([0, 0, 100])
    upper_gray = np.array([180, 30, 250])
    mask = cv2.inRange(hsv, lower_gray, upper_gray)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def filter_runway_lines(lines, image_shape):
    """Filter lines based on runway-specific criteria"""
    if lines is None:
        return []

    filtered_lines = []
    img_height, img_width = image_shape[:2]
    min_length = img_width * 0.15  # Runway should be at least 15% of image width

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculate line properties
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

        # Runway-specific criteria:
        # 1. Long enough
        # 2. Nearly horizontal or vertical (within 20 degrees)
        # 3. Not too close to image edges
        if (
            length >= min_length
            and (angle <= 20 or abs(angle - 90) <= 20 or abs(angle - 180) <= 20)
            and min(x1, x2) > img_width * 0.05
            and max(x1, x2) < img_width * 0.95
        ):
            filtered_lines.append(line)

    return filtered_lines


def cluster_similar_lines(filtered_lines):
    """Cluster similar lines to identify main runway direction"""
    if not filtered_lines:
        return []

    # Extract angles and lengths for clustering
    line_features = []
    for line in filtered_lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        line_features.append([angle % 180, length])

    # Cluster lines
    clustering = DBSCAN(eps=10, min_samples=2).fit(line_features)

    # Find the largest cluster
    if len(set(clustering.labels_)) <= 1:
        return filtered_lines

    labels = clustering.labels_
    largest_cluster = max(set(labels) - {-1}, key=lambda x: list(labels).count(x))

    # Return lines from the largest cluster
    return [
        line for i, line in enumerate(filtered_lines) if labels[i] == largest_cluster
    ]


def detect_runway(image, params, debug=False):
    """Main function to detect runway"""
    runway_mask = create_runway_mask(image)
    if debug:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(5, 1, figsize=(24, 56))
        axes[0].imshow(runway_mask, cmap="gray")
        axes[0].set_title("Runway Mask")

    preprocessed = preprocess_image(image)
    if debug:
        axes[1].imshow(preprocessed)
        axes[1].set_title("Preprocessed Image")

    preprocessed = cv2.bitwise_and(preprocessed, preprocessed, mask=runway_mask)
    if debug:
        axes[2].imshow(preprocessed)
        axes[2].set_title("Masked Preprocessed Image")

    edges = cv2.Canny(
        preprocessed, params["canny_low"], params["canny_high"], apertureSize=3
    )
    if debug:
        axes[3].imshow(preprocessed, cmap="gray")
        axes[3].set_title("Canny Edges")

    lines = cv2.HoughLinesP(
        edges,
        rho=params["rho"],
        theta=params["theta"],
        threshold=params["hough_threshold"],  # Increased threshold
        minLineLength=params["min_line_length"],
        maxLineGap=params["max_line_gap"],  # Reduced gap
    )

    # Filter and cluster lines
    filtered_lines = filter_runway_lines(lines, image.shape)
    final_lines = cluster_similar_lines(filtered_lines)

    if debug:
        # Draw results
        result = image.copy()
        if final_lines:
            for line in final_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

        axes[4].imshow(result)
        axes[4].set_title("Detected Runway")
        plt.savefig("./line-debug.png")
        plt.show()

    return final_lines
