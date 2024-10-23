from torchvision.io import read_image
from torchvision.transforms import ToTensor
import cv2

im1 = cv2.imread("./noi-bai-01.tif", cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread("./noi-bai-02.tif", cv2.IMREAD_GRAYSCALE)

im1 = cv2.equalizeHist(im1)
im2 = cv2.equalizeHist(im2)

fft_diff = gen_fft_diff_mask(im1, im2, window_size=64, window_stride=32)

mask_img = mask2image(fft_diff)
cv2.imwrite("./mask.png", mask_img)

plt.imshow(mask_img, cmap="gray", interpolation="nearest")
plt.colorbar()  # Add color bar for reference
plt.title("Normalized Matrix as Image")
plt.show()
