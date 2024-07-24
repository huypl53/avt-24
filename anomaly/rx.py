import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure
import imageio
import sys
import os


im_path = sys.argv[1]


# RX Detector function with regularization
def rx_detector(data, reg_value=1e-6):
    mean_vec = np.mean(data, axis=0)
    centered_data = data - mean_vec
    cov_matrix = np.cov(centered_data, rowvar=False)

    # Regularize the covariance matrix by adding reg_value to the diagonal elements
    cov_matrix += reg_value * np.eye(cov_matrix.shape[0])

    inv_cov_matrix = np.linalg.inv(cov_matrix)
    rx_scores = np.array(
        [np.dot(np.dot(sample.T, inv_cov_matrix), sample) for sample in centered_data]
    )
    return rx_scores


# Load RGB image
image = io.imread(im_path)

# Check image dimensions
height, width, channels = image.shape

# Flatten the spatial dimensions for RX Detector
flattened_data = image.reshape(-1, channels).astype(np.float32)

# Run RX detector
rx_scores = rx_detector(flattened_data)

# Reshape RX scores back to the spatial dimensions
rx_image = rx_scores.reshape(height, width)

# Determine a threshold for anomalies (e.g., top 5% of the RX scores)
threshold = np.percentile(rx_scores, 95)

# Create a binary mask of anomalies
anomaly_mask = rx_image > threshold

# Detect contours in the anomaly mask
contours = measure.find_contours(anomaly_mask, level=0.5)

bname = os.path.basename(im_path)
# Save the contours to a .txt file
contours_filename = bname.split(".", 1)[0] + ".txt"
with open(contours_filename, "w") as f:
    for contour in contours:
        for point in contour:
            f.write(f"{int(point[1])} {int(point[0])} ")
        f.write("\n")

# Save the anomaly mask to a file
anomaly_mask_filename = "_mask.".join(bname.rsplit(".", 1))
imageio.imwrite(anomaly_mask_filename, (anomaly_mask * 255).astype(np.uint8))

# # Plot the original image
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(image)
# plt.axis("off")
#
# # Plot the anomaly mask
# plt.subplot(1, 2, 2)
# plt.title("Anomaly Detection")
# plt.imshow(image)
# for contour in contours:
#     plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
# plt.axis("off")
#
# plt.show()

print(f"Anomaly mask saved to {anomaly_mask_filename}")
print(f"Contours saved to {contours_filename}")
