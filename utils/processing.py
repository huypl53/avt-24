from typing import List
import cv2
import numpy as np

def find_boundary_keypoints(binary_image) -> List[np.ndarray]:
    '''
    return keypoints: List[Tuple[int, int]] in (x, y) format
    '''
    # Ensure the image is in binary format (values 0 or 255)
    binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)[1]
    
    # Find contours of the areas in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract keypoints from the contours
    keypoints = [contour.squeeze() for contour in contours if len(contour) > 2]  # Ignore single-point contours

    return keypoints