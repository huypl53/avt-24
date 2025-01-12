import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, TypeVar, Union

import numpy as np
import rasterio
from rasterio.errors import RasterioError
from rasterio.warp import transform

from core.multi_processing.parallel import ParallelProcessor


class RasterImageError(Exception):
    """Custom exception for RasterImage-specific errors"""

    pass


T = TypeVar("T")
R = TypeVar("R")


def _convert_coordinates_chunk(
    transform_data: dict, coords: List[Tuple[int, int]]
) -> List[Tuple[float, float]]:
    """Static method to convert coordinates using just the transform data"""
    from rasterio.transform import xy

    results = []
    transform = transform_data["transform"]
    for x, y in coords:
        lon, lat = xy(transform, y, x)
        results.append((lat, lon))
    return results


class RasterImage(ParallelProcessor[Tuple[int, int], Tuple[float, float]]):
    def __init__(self, source: Union[str, Path, bytes], max_workers: int = None):
        """
        Initialize RasterImage with either a file path or bytes data

        Args:
            source: Path to the TIFF file or bytes containing TIFF data
            max_workers: Number of worker processes for parallel processing
        """
        super().__init__(max_workers or multiprocessing.cpu_count())
        self._source = source
        self._dataset = None
        self._numpy_data = None
        self._bounds = None
        self._transform = None
        self._crs = None

        self._load_dataset()
        self._load_metadata()

    def _load_dataset(self):
        """Load the raster dataset from the source"""
        try:
            if isinstance(self._source, (str, Path)):
                self._dataset = rasterio.open(self._source)
            elif isinstance(self._source, bytes):
                self._dataset = rasterio.MemoryFile(self._source).open()
            else:
                raise RasterImageError("Source must be either a path or bytes")
        except RasterioError as e:
            raise RasterImageError(f"Failed to load raster data: {str(e)}")

    def _load_metadata(self):
        """Load and store the geospatial metadata"""
        self._bounds = self._dataset.bounds
        self._transform = self._dataset.transform
        self._crs = self._dataset.crs

    @property
    def numpy(self) -> np.ndarray:
        """
        Get the image data as a numpy array

        Returns:
            np.ndarray: The image data
        """
        if self._numpy_data is None:
            image_data = self._dataset.read()
            image_data = np.transpose(image_data, (1, 2, 0))
            self._numpy_data = image_data
        return self._numpy_data

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get the image bounds (left, bottom, right, top)"""
        return self._bounds

    @property
    def transform(self):
        """Get the affine transform"""
        return self._transform

    @property
    def crs(self):
        """Get the coordinate reference system"""
        return self._crs

    def contains_bounds(self, other_bounds: Tuple[float, float, float, float]) -> bool:
        """
        Check if the given bounds are completely inside this image
        Args:
            other_bounds: Tuple of (left, bottom, right, top) coordinates

        Returns:
            bool: True if the other bounds are completely inside this image
        """
        left, bottom, right, top = other_bounds
        return (
            left >= self._bounds.left
            and right <= self._bounds.right
            and bottom >= self._bounds.bottom
            and top <= self._bounds.top
        )

    @classmethod
    def find_intersection(
        cls, raster_a: "RasterImage", rasters: List["RasterImage"]
    ) -> "RasterImage":
        """
        Find the intersection of multiple RasterImages that is completely inside raster_a

        Args:
            raster_a: The reference RasterImage
            rasters: List of RasterImages to find intersection with

        Returns:
            RasterImage: A new RasterImage representing the intersection

        Raises:
            RasterImageError: If the intersection is not completely inside raster_a
        """
        if not rasters:
            raise RasterImageError("No rasters provided for intersection")

        left = max(r._bounds.left for r in [raster_a] + rasters)
        bottom = max(r._bounds.bottom for r in [raster_a] + rasters)
        right = min(r._bounds.right for r in [raster_a] + rasters)
        top = min(r._bounds.top for r in [raster_a] + rasters)

        intersection_bounds = (left, bottom, right, top)

        if left >= right or bottom >= top:
            raise RasterImageError("No valid intersection found")

        if not raster_a.contains_bounds(intersection_bounds):
            raise RasterImageError("Intersection is not completely inside first image")

        window = rasterio.windows.from_bounds(
            *intersection_bounds, transform=raster_a._transform
        )
        intersection_data = raster_a._dataset.read(window=window)
        profile = raster_a._dataset.profile.copy()
        profile.update(
            {
                "height": window.height,
                "width": window.width,
                "transform": rasterio.windows.transform(window, raster_a._transform),
            }
        )

        with BytesIO() as intersection_bytes:
            with rasterio.open(intersection_bytes, "w", **profile) as dataset:
                dataset.write(intersection_data)
            intersection_raster = cls(intersection_bytes.getvalue())

        return intersection_raster

    def crop_raster(self, source: "RasterImage") -> Tuple["RasterImage", bool]:

        left, bottom, right, top = source.bounds
        if self.contains_bounds((left, bottom, right, top)):
            window = rasterio.windows.from_bounds(
                left, bottom, right, top, transform=self.transform
            )
            cropped_data = self._dataset.read(window=window)
            new_transform = rasterio.windows.transform(window, self.transform)
            new_profile = self._dataset.profile
            new_profile.update(
                {
                    "height": cropped_data.shape[1],
                    "width": cropped_data.shape[2],
                    "transform": new_transform,
                }
            )

            with BytesIO() as bytes_data:
                with rasterio.open(bytes_data, "w", **new_profile) as dataset:
                    dataset.write(cropped_data)
                cropped_raster = RasterImage(bytes_data.getvalue())
                return cropped_raster, True
        else:
            return self, False

    def pixel_to_coords(self, x: int, y: int) -> Tuple[float, float]:
        """Converts pixel coordinates (x, y) to latitude and longitude."""
        easting, northing = rasterio.transform.xy(self.transform, y, x, offset="center")
        if self.crs.to_string() == "EPSG:4326":
            return northing, easting  # Since (lat, lon) = (y, x)

        lon, lat = transform(self.crs, "EPSG:4326", [easting], [northing])
        return lat[0], lon[0]

    def replace_image_data(self, cv2_image: np.ndarray):
        """
        Replace the original image data with a cv2 image

        Args:
            cv2_image (np.ndarray): OpenCV image to replace the original data
                                    Expected to be in (height, width, channels) format

        Raises:
            RasterImageError: If the input image dimensions or type are incompatible
        """
        if len(cv2_image.shape) == 2:  # Grayscale
            image_data = cv2_image[np.newaxis, ...]
        elif (
            len(cv2_image.shape) == 3 and cv2_image.shape[2] <= 4
        ):  # Color (RGB or RGBA)
            image_data = cv2_image.transpose(
                2, 0, 1
            )  # Rearrange to [channels, height, width]
        else:
            raise RasterImageError("Invalid image format for raster data.")

        self._numpy_data = cv2_image
        profile = self._dataset.profile.copy()

        num_channels = cv2_image.shape[2]
        profile.update(
            {
                "height": cv2_image.shape[0],
                "width": cv2_image.shape[1],
                "count": num_channels,
                "dtype": self._numpy_data.dtype,
            }
        )

        with BytesIO() as bytes_data:
            with rasterio.open(bytes_data, "w", **profile) as dataset:
                dataset.write(image_data)
            self._dataset = rasterio.MemoryFile(bytes_data.getvalue()).open()

    def to_bytes(self):
        """
        Convert the current raster dataset to bytes.

        Returns:
            bytes: The TIFF file data as bytes.
        """
        from rasterio.io import MemoryFile

        if not self._dataset:
            raise RasterImageError("No dataset available to convert to bytes.")

        # Write the dataset to a MemoryFile and return as bytes
        with MemoryFile() as memfile:
            with memfile.open(**self._dataset.profile) as dst:
                dst.write(self._numpy_data.transpose(2, 0, 1))
            return memfile.read()

    def _convert_single_coordinate(self, coord: Tuple[int, int]) -> Tuple[float, float]:
        """
        Convert a single pixel coordinate to lat/lon coordinate

        Args:
            coord: (x, y) pixel coordinate

        Returns:
            (lat, lon) coordinate
        """
        return self.pixel_to_coords(*coord)

    def _get_transform_data(self):
        """Get the essential transformation data needed for coordinate conversion"""
        return {
            "transform": self._transform,
            "crs": self._crs,
        }

    def process_coordinates_parallel(
        self, keypoints: List[Tuple[int, int]], batch_size: int = 10000
    ) -> List[Tuple[float, float]]:
        """
        Convert pixel coordinates to lat/lon coordinates in parallel, processing in batches

        Args:
            keypoints: List of (x, y) pixel coordinates
            batch_size: Number of keypoints to process in each batch

        Returns:
            List of (lat, lon) coordinates
        """

        # Extract just the necessary transformation data
        transform_data = self._get_transform_data()

        # Create a partial function with the transform data
        convert_func = partial(_convert_coordinates_chunk, transform_data)

        # Split keypoints into batches
        batches = [
            keypoints[i : i + batch_size] for i in range(0, len(keypoints), batch_size)
        ]

        # Process batches in parallel
        results = []
        with ProcessPoolExecutor() as executor:
            for batch_results in executor.map(convert_func, batches):
                results.extend(batch_results)

        return results

    def process_coordinates_list_parallel(
        self, keypoints: List[List[Tuple[int, int]]]
    ) -> List[List[Tuple[float, float]]]:
        """
        Convert pixel coordinates to lat/lon coordinates in parallel

        Args:
            keypoints: List of lists of (x, y) pixel coordinates

        Returns:
            List of lists of (lat, lon) coordinates
        """
        return self.process_parallel(keypoints, self._convert_keypoint_list)

    def _convert_keypoint_list(
        self, keypoint_list: List[Tuple[int, int]]
    ) -> List[Tuple[float, float]]:
        """
        Convert a list of pixel coordinates to lat/lon coordinates

        Args:
            keypoint_list: List of (x, y) pixel coordinates

        Returns:
            List of (lat, lon) coordinates
        """
        return [self.pixel_to_coords(x, y) for x, y in keypoint_list]


if __name__ == "__main__":
    file_paths = [
        "Katterbach Kaserne-01.tif",
        "Katterbach Kaserne-02.tif",
    ]  # "Katterbach Kaserne-02.tif", "Katterbach Kaserne-03.tif"]
    raster_images = [RasterImage(fp) for fp in file_paths]
    raster_image = RasterImage.find_intersection(raster_images[0], raster_images[1:])

    print(raster_image.numpy.shape)
