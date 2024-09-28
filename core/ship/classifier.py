import os

import cv2
import joblib
import numpy as np
from skimage.feature import hog

from utils.path import main_script_dir

SHIP_CLASSIFICATION_MODEL_PATH = os.path.join(
    main_script_dir, "weights", "ship_classification.joblib"
)
SHIP_CLASSIFICATION_SCALER_PATH = os.path.join(
    main_script_dir, "weights", "ship_scaler.joblib"
)
model = joblib.load(SHIP_CLASSIFICATION_MODEL_PATH)
scaler = joblib.load(SHIP_CLASSIFICATION_SCALER_PATH)


def extract_features(image_path, width, height):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (50, 25), interpolation=cv2.INTER_AREA)
    hog_feature = hog(
        resized_image,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
    )
    combined_features = np.concatenate([[width * 0.8, height * 0.8], hog_feature * 0.2])
    return combined_features


class Classification_Image:
    def __init__(self):
        pass

    def classify(self, src_img_path, width, height):
        features = extract_features(src_img_path, width, height)
        features = scaler.transform([features])
        prediction = model.predict(features)
        return prediction[0]


def extract_ship_feature(im: np.ndarray) -> np.ndarray:
    assert im.ndim == 3
    if im.shape[-1] == 1:
        pass
    else:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    height, width = im.shape[:2]

    resized_image = cv2.resize(im, (50, 25), interpolation=cv2.INTER_AREA)
    hog_feature = hog(
        resized_image,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
    )
    combined_features = np.concatenate([[width * 0.8, height * 0.8], hog_feature * 0.2])
    return combined_features


def classify_ship(im: np.ndarray):
    features = extract_ship_feature(im)
    features = scaler.transform([features])
    prediction = model.predict(features)
    return prediction[0]
