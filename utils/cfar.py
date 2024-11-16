import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel
from tqdm import tqdm

from core.multi_processing.parallel import ParallelProcessor


# @dataclass
class CFARParams(BaseModel):
    """Parameters for CFAR detection"""

    guard_cells: Optional[Tuple[int, int]]  # Number of guard cells (rows, cols)
    training_cells: Optional[Tuple[int, int]]  # Number of training cells (rows, cols)
    false_alarm_rate: Optional[float]  # Desired false alarm rate
    scaling_factor: Optional[float] = 1.5  # Direct scaling factor for threshold
    min_training_cells: Optional[int] = 1  # Minimum number of training cells required


class CFAR2D(ParallelProcessor):
    """2D Constant False Alarm Rate (CFAR) detector for signal processing."""

    def __init__(self, params: CFARParams, *args, **kwargs):
        self.params = params
        super().__init__(*args, **kwargs)

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

    def _process_chunk(self, args) -> Tuple[np.ndarray, np.ndarray, slice]:
        """Process a chunk of rows"""
        matrix, row_slice, stride = args
        rows = row_slice.stop - row_slice.start
        cols = matrix.shape[1] - 2 * self.params.training_cells[1]  # Adjust for padding

        chunk_threshold = np.zeros((rows, cols), dtype=float)
        chunk_detections = np.zeros((rows, cols), dtype=bool)

        tr, tc = self.params.training_cells

        for i in range(0, rows, stride):
            for j in range(0, cols, stride):
                training_cells = self._get_training_cells(
                    matrix,
                    i + tr,  # Adjust for padding offset
                    j + tc,  # Adjust for padding offset
                )

                if len(training_cells) >= self.params.min_training_cells:
                    threshold = np.mean(training_cells) * self.params.scaling_factor

                    end_i = min(i + stride, rows)
                    end_j = min(j + stride, cols)
                    try:
                        chunk_threshold[i:end_i, j:end_j] = threshold
                        left_slice = chunk_detections[i:end_i, j:end_j]
                        right_slice = matrix[
                            i + tr : i + tr + (end_i - i),
                            j + tc : j + tc + (end_j - j),
                        ]
                        print(
                            f"Left shape: {left_slice.shape}, Right shape: {right_slice.shape}"
                        )
                        chunk_detections[i:end_i, j:end_j] = right_slice > threshold
                    except Exception as e:
                        print(
                            f"Left shape: {left_slice.shape}, Right shape: {right_slice.shape}"
                        )

                        print(f"Error processing chunk: {e}")
                        print(
                            f"i={i}, end_i={end_i}, j={j}, end_j={end_j}, tr={tr}, tc={tc}"
                        )

        return chunk_threshold, chunk_detections, row_slice

    def apply(
        self, matrix: np.ndarray, stride: int = 1, n_processes: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply CFAR detection using ProcessPoolExecutor
        Args:
            matrix: Input matrix
            stride: Step size for processing (default=1)
            n_processes: Number of processes to use (defaults to self.max_workers)
        """
        tr, tc = self.params.training_cells
        max_training_size = max(tr, tc)

        # Validate and adjust stride if necessary
        if stride > max_training_size:
            original_stride = stride
            stride = max_training_size
            warnings.warn(
                f"Stride ({original_stride}) is larger than training cell size ({max_training_size}). "
                f"Adjusting stride to {stride} to avoid missing detections."
            )

        n_processes = n_processes or self.max_workers
        rows, cols = matrix.shape

        # Add padding to handle edge cases
        padded_matrix = np.pad(matrix, ((tr, tr), (tc, tc)), mode="reflect")

        # Create smaller chunks to reduce memory usage
        chunk_size = min(1000, rows // n_processes)  # Limit chunk size
        threshold_matrix = np.zeros_like(matrix, dtype=float)
        detections = np.zeros_like(matrix, dtype=bool)

        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            # Process chunks in batches to control memory usage
            for start in tqdm(
                range(0, rows, chunk_size * n_processes), desc="Processing batches"
            ):
                futures = {}

                # Submit a batch of chunks
                for i in range(
                    start, min(start + chunk_size * n_processes, rows), chunk_size
                ):
                    row_slice = slice(i, min(i + chunk_size, rows))
                    # Extract only the needed portion of padded_matrix
                    chunk_data = padded_matrix[
                        i : i + chunk_size + 2 * tr, :
                    ]  # Include padding
                    future = executor.submit(
                        self._process_chunk, (chunk_data, row_slice, stride)
                    )
                    futures[future] = row_slice

                # Process completed futures for this batch
                for future in as_completed(futures):
                    row_slice = futures[future]
                    chunk_threshold, chunk_detections, _ = future.result()
                    threshold_matrix[row_slice] = chunk_threshold
                    detections[row_slice] = chunk_detections

        return threshold_matrix, detections

    def get_top_detections(
        self,
        matrix: np.ndarray,
        n_top: int = None,
        min_threshold: float = None,
        stride=2,
    ) -> List[Tuple[int, int, float]]:
        """
        Get top N detections sorted by value
        """
        _, detections = self.apply(matrix, stride=stride)

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
