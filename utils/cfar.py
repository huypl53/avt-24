from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel


# @dataclass
class CFARParams(BaseModel):
    """Parameters for CFAR detection"""

    guard_cells: Optional[Tuple[int, int]]  # Number of guard cells (rows, cols)
    training_cells: Optional[Tuple[int, int]]  # Number of training cells (rows, cols)
    false_alarm_rate: Optional[float]  # Desired false alarm rate
    scaling_factor: Optional[float] = 1.5  # Direct scaling factor for threshold
    min_training_cells: Optional[int] = 1  # Minimum number of training cells required


class CFAR2D:
    def __init__(self, params: CFARParams):
        self.params = params

    def _get_training_cells(
        self, matrix: np.ndarray, center_row: int, center_col: int
    ) -> np.ndarray:
        """
        Extract training cells around the CUT (Cell Under Test)
        Excluding guard cells and CUT itself
        """
        rows, cols = matrix.shape
        gr, gc = self.params.guard_cells
        tr, tc = self.params.training_cells

        # Calculate window boundaries
        row_start = max(0, center_row - tr)
        row_end = min(rows, center_row + tr + 1)
        col_start = max(0, center_col - tc)
        col_end = min(cols, center_col + tc + 1)

        # Get the full window
        window = matrix[row_start:row_end, col_start:col_end].copy()

        # Create a mask for guard cells and CUT
        guard_mask = np.ones_like(window, dtype=bool)
        gr_start = max(0, tr - center_row + gr)
        gr_end = min(window.shape[0], tr - center_row + 2 * gr + 1)
        gc_start = max(0, tc - center_col + gc)
        gc_end = min(window.shape[1], tc - center_col + 2 * gc + 1)

        guard_mask[gr_start:gr_end, gc_start:gc_end] = False

        return window[guard_mask]

    def apply(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply CFAR detection to the input matrix
        Returns:
            - threshold_matrix: Matrix of calculated thresholds
            - detections: Binary matrix indicating detections
        """
        rows, cols = matrix.shape
        threshold_matrix = np.zeros_like(matrix, dtype=float)
        detections = np.zeros_like(matrix, dtype=bool)

        # Add padding to handle edge cases
        tr, tc = self.params.training_cells
        padded_matrix = np.pad(matrix, ((tr, tr), (tc, tc)), mode="reflect")

        for i in range(rows):
            for j in range(cols):
                # Get training cells
                training_cells = self._get_training_cells(
                    padded_matrix,
                    i + tr,  # Adjust for padding
                    j + tc,  # Adjust for padding
                )

                if len(training_cells) >= self.params.min_training_cells:
                    # Calculate threshold using mean and scaling factor
                    threshold = np.mean(training_cells) * self.params.scaling_factor
                    threshold_matrix[i, j] = threshold

                    # Compare CUT with threshold
                    if matrix[i, j] > threshold:
                        detections[i, j] = True

        return threshold_matrix, detections

    def get_top_detections(
        self, matrix: np.ndarray, n_top: int = None, min_threshold: float = None
    ) -> List[Tuple[int, int, float]]:
        """
        Get top N detections sorted by value
        """
        _, detections = self.apply(matrix)

        # Get all detection coordinates and values
        detection_coords = np.where(detections)
        detection_values = matrix[detection_coords]

        # Create list of (row, col, value) tuples
        detections_list = list(
            zip(detection_coords[0], detection_coords[1], detection_values)
        )

        # Sort by value in descending order
        detections_list.sort(key=lambda x: x[2], reverse=True)

        # Apply minimum threshold if specified
        if min_threshold is not None:
            detections_list = [d for d in detections_list if d[2] >= min_threshold]

        # Return top N if specified
        if n_top is not None:
            return detections_list[:n_top]

        return detections_list
