import numpy as np
from typing import Callable, Tuple, List, Literal
import cv2


class SlidingWindowInference:
    def __init__(
        self,
        inference_fn: Callable[[np.ndarray], np.ndarray],
        window_size: Tuple[int, int] = (512, 512),
        overlap: float = 0.25,
        batch_size: int = 2,
        smoothier: bool = True,
    ):
        """
        Initialize sliding window inference.

        Args:
            inference_fn: A callable that takes an input image and returns a 2D segmentation mask
            window_size: Size of sliding window (height, width)
            overlap: Overlap between windows (0-1)
            batch_size: Batch size for inference
        """
        self.inference_fn = inference_fn
        self.window_size = window_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.smoothier = smoothier

    def _get_windows(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Extract windows from large image with overlap."""
        h, w = image.shape[:2]
        window_h, window_w = self.window_size

        # Adjust window size if it's larger than the image
        window_h = min(window_h, h)
        window_w = min(window_w, w)

        # Calculate stride (step size)
        stride_h = int(window_h * (1 - self.overlap))
        stride_w = int(window_w * (1 - self.overlap))

        # Calculate number of windows needed
        n_h = max(1, int(np.ceil((h - window_h) / stride_h) + 1))
        n_w = max(1, int(np.ceil((w - window_w) / stride_w) + 1))

        windows = []
        positions = []

        for i in range(n_h):
            for j in range(n_w):
                # Calculate start positions
                start_h = min(i * stride_h, h - window_h)
                start_w = min(j * stride_w, w - window_w)

                # Ensure start positions are non-negative
                start_h = max(0, start_h)
                start_w = max(0, start_w)

                # Extract window
                end_h = start_h + window_h
                end_w = start_w + window_w

                window = image[start_h:end_h, start_w:end_w]
                windows.append(window)
                positions.append((start_h, start_w))

                # print(f"Window {i},{j}: pos=({start_h},{start_w}), shape={window.shape}")  # Debug info

        return windows, positions

    def _merge_predictions(
        self,
        predictions: List[np.ndarray],
        positions: List[Tuple[int, int]],
        original_size: Tuple[int, int],
    ) -> np.ndarray:
        """Merge predictions by placing them in their original positions."""
        h, w = original_size
        final_prediction = np.zeros((h, w), dtype=np.float32)

        for pred, (start_h, start_w) in zip(predictions, positions):
            end_h = start_h + pred.shape[0]
            end_w = start_w + pred.shape[1]

            # Handle edge cases when the window is larger than the image
            overlap_h = max(0, min(end_h, h) - max(start_h, 0))
            overlap_w = max(0, min(end_w, w) - max(start_w, 0))

            pred_area = pred[:overlap_h, :overlap_w]
            if self.smoothier:
                pred_area = smooth_runway_lines(pred_area)
            # Copy the prediction to the final mask, handling overlap
            final_prediction[
                max(0, start_h) : max(0, start_h) + overlap_h,
                max(0, start_w) : max(0, start_w) + overlap_w,
            ] = pred_area

        return final_prediction

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Perform inference on large image using sliding window approach.

        Args:
            image: Input image (H, W, C)

        Returns:
            Segmentation mask (H, W)
        """
        # Extract windows and their positions
        windows, positions = self._get_windows(image)

        # Process windows in batches
        # predictions = [self.inference_fn(window) for window in windows]
        predictions = []
        for window, position in zip(windows, positions):
            prediction = self.inference_fn(window)
            predictions.append(prediction)
            # print(window.shape, position, prediction.shape)

        # Merge predictions
        final_prediction = self._merge_predictions(
            predictions, positions, image.shape[:2]
        )

        return final_prediction


def smooth_runway_lines(binary_mask):
    # binary_mask = (binary_mask > 0).astype(np.uint8) * 255
    binary_mask = cv2.normalize(binary_mask, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    cleaned_uint8 = np.uint8(cleaned)

    lines = cv2.HoughLinesP(
        cleaned_uint8,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=100,
        maxLineGap=50,
    )

    result = np.zeros_like(binary_mask)

    # Draw detected lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), 255, 2)

    return result


if __name__ == "__main__":

    import cv2
    from mmseg.apis import init_segmentor, inference_segmentor

    config_file = "/workspace/mmsegmentation/work_dirs/runway_config/runway_config.py"
    checkpoint_file = "/workspace/mmsegmentation/work_dirs/runway_config/latest.pth"
    model = init_segmentor(config_file, checkpoint_file, device="cuda:0")
    # img_path = '/workspace/data/runway/test/images/RUNWAY_Artificial_island_Hong_Kong_International_Airport_2024-11-07_14-10-34-737_36_3m.tif'
    img_path = "/workspace/data/runway/test/images/RUNWAY_Artificial_island_Dalian_Zhoushuizi_International_Airport_2024-11-07_14-05-16-952_35_1m.tif"
    img = cv2.imread(img_path)

    def infer_image(img):

        result = inference_segmentor(model, img)
        # print(f'Model result shape: {result[0].shape}')
        return result[0]

    slicer = SlidingWindowInference(inference_fn=infer_image, window_size=(1024, 1024))
    result = slicer(img)

    # result = infer_image(img)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

    # print(f'final result shape: {result.shape}')
    cv2.imwrite("./seg-result.png", result)

    #################
    # img_path = '/workspace/mmsegmentation/demo/RUNWAY_Artificial_island_Hong_Kong_International_Airport_2024-11-07_14-10-34-737_36_3m-result.png'
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # smooth_img = smooth_runway_lines(img)

    # cv2.imwrite('./smooth.png', smooth_img)
