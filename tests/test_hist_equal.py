import sys

import cv2

from enhancing import hist_equalize

im_path = sys.argv[1]
im = cv2.imread(im_path)
h, w = im.shape[:2]
n_h, n_w = int(1024 / w * h), 1024

im_small = cv2.resize(im, (n_w, n_h))
cv2.imshow("origin", im_small)

enhanced_im = hist_equalize(im)
enhanced_im_small = cv2.resize(enhanced_im, (n_w, n_h))
# cv2.imshow("enhanced", enhanced_im_small)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
