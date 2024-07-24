import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import sys

im_path = sys.argv[1]
lb_path = sys.argv[2]


def read_label_file(filename):
    contours = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip():  # Ensure it's not an empty line
                points = line.strip().split()
                contour = [
                    (int(points[i]), int(points[i + 1]))
                    for i in range(0, len(points), 2)
                ]
                contours.append(np.array(contour))
    return contours


# Load the original image
image = io.imread(im_path)

# Read the contours from the label file
contours = read_label_file(lb_path)

# Define a colormap to get unique colors
colormap = plt.cm.get_cmap("tab20", len(contours))

# Plot the original image
plt.figure(figsize=(12, 6))
plt.imshow(image)
plt.title("Labeled Image")

# Draw the contours on the image
for i, contour in enumerate(contours):
    color = colormap(i)
    plt.plot(contour[:, 0], contour[:, 1], linewidth=2, color=color)

plt.axis("off")
plt.show()
