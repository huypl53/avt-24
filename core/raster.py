import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.errors import RasterioError
from typing import Union, List, Tuple, Optional
from pathlib import Path
from io import BytesIO
from rasterio.warp import transform

class RasterImageError(Exception):
    """Custom exception for RasterImage-specific errors"""
    pass

class RasterImage:
    def __init__(self, source: Union[str, Path, bytes]):
        """
        Initialize RasterImage with either a file path or bytes data
        
        Args:
            source: Path to the TIFF file or bytes containing TIFF data
        """
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
                # Use rasterio's MemoryFile for bytes input
                # with rasterio.MemoryFile(self._source) as memfile:
                #     self._dataset = memfile.open()
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
        return (left >= self._bounds.left and right <= self._bounds.right and
                bottom >= self._bounds.bottom and top <= self._bounds.top)
    
    @classmethod
    def find_intersection(cls, raster_a: 'RasterImage', rasters: List['RasterImage']) -> 'RasterImage':
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
        
        # Start with the bounds of the first raster
        left = max(r._bounds.left for r in [raster_a] + rasters)
        bottom = max(r._bounds.bottom for r in [raster_a] + rasters)
        right = min(r._bounds.right for r in [raster_a] + rasters)
        top = min(r._bounds.top for r in [raster_a] + rasters)
        
        intersection_bounds = (left, bottom, right, top)
        
        # Check if intersection is valid
        if left >= right or bottom >= top:
            raise RasterImageError("No valid intersection found")
            
        # Check if intersection is inside raster_a
        if not raster_a.contains_bounds(intersection_bounds):
            raise RasterImageError("Intersection is not completely inside first image")
        
        # Create a new raster for the intersection
        # First, we need to get the pixel coordinates for the intersection
        window = rasterio.windows.from_bounds(*intersection_bounds, 
                                            transform=raster_a._transform)
        
        # Read the data for the intersection window
        intersection_data = raster_a._dataset.read(window=window)
        
        # Create a new dataset with the intersection
        profile = raster_a._dataset.profile.copy()
        profile.update({
            'height': window.height,
            'width': window.width,
            'transform': rasterio.windows.transform(window, raster_a._transform)
        })
        
        # Create a BytesIO object to store the intersection data
        with BytesIO() as intersection_bytes:
            with rasterio.open(intersection_bytes, 'w', **profile) as dataset:
                dataset.write(intersection_data)
            
            # Create a new RasterImage from the intersection data
            intersection_raster = cls(intersection_bytes.getvalue())
        
        return intersection_raster
    
    def crop_raster(self, source: 'RasterImage') -> Tuple['RasterImage', bool]:
        # Get the bounds of the intersection
        left, bottom, right, top = source.bounds
        
        # Check if the intersection is fully contained within the current RasterImage
        if self.contains_bounds((left, bottom, right, top)):
            # Calculate the window for the intersection
            window = rasterio.windows.from_bounds(left, bottom, right, top, transform=self.transform)
            
            # Read the data for the intersection window
            # cropped_data = self.numpy[
            #     window.out_shape[0]:window.out_shape[1], 
            #     window.out_shape[2]:window.out_shape[3]
            # ]
            
            # # Create a new RasterImage with the cropped data
            # cropped_raster = RasterImage(BytesIO(cropped_data.tobytes()))
            # cropped_raster._bounds = source.bounds
            # cropped_raster._transform = source.transform
            # cropped_raster._crs = source.crs
            
            # return cropped_raster, True

            # Read the data for all bands within this window
            cropped_data = self._dataset.read(window=window)

            # Define new metadata
            new_transform = rasterio.windows.transform(window, self.transform)
            new_profile = self._dataset.profile
            new_profile.update({
                "height": cropped_data.shape[1],
                "width": cropped_data.shape[2],
                "transform": new_transform,
            })

            # # Save cropped data to an in-memory file and return a new RasterImage instance
            # memfile = rasterio.MemoryFile()
            # with memfile.open(**new_profile) as dataset:
            #     dataset.write(cropped_data)
            #     # Return a new RasterImage created from the in-memory dataset
            #     return RasterImage(byte_data=memfile.read())

            with BytesIO() as bytes_data:
                with rasterio.open(bytes_data, 'w', **new_profile) as dataset:
                    dataset.write(cropped_data)
                
                # Create a new RasterImage from the intersection data
                cropped_raster = RasterImage(bytes_data.getvalue())
                return cropped_raster, True
        
        else:
            # If the intersection is not fully contained, return the original RasterImage
            return self, False
    
    def pixel_to_coords(self, x: int, y: int) -> Tuple[float, float]:
        """Converts pixel coordinates (x, y) to latitude and longitude."""
        # Get easting and northing using the image transform
        easting, northing = rasterio.transform.xy(self.transform, y, x, offset="center")

        # If the image CRS is already in WGS84, return directly
        if self.crs.to_string() == "EPSG:4326":
            return northing, easting  # Since (lat, lon) = (y, x)

        # Otherwise, reproject to WGS84
        lon, lat = transform(self.crs, "EPSG:4326", [easting], [northing])
        return lat[0], lon[0]

if __name__ == '__main__':
    file_paths = ["Katterbach Kaserne-01.tif", "Katterbach Kaserne-02.tif"] #"Katterbach Kaserne-02.tif", "Katterbach Kaserne-03.tif"]
    raster_images = [RasterImage(fp) for fp in file_paths]
    raster_image = RasterImage.find_intersection(raster_images[0], raster_images[1:])
    
    print(raster_image.numpy.shape)