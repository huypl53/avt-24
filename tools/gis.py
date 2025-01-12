import requests
import rasterio
from rasterio.transform import from_bounds
import numpy as np
from PIL import Image
import io

image_service_url = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export"


def download_arcgis(bbox, output_path, token=None):
    """
    Download and properly georeference imagery from ArcGIS Online

    Parameters:
    bbox (list): Bounding box coordinates [xmin, ymin, xmax, ymax] in WGS84
    output_path (str): Path where the georeferenced TIFF will be saved
    token (str): Optional ArcGIS Online token
    """

    # ArcGIS World Imagery service URL

    # Calculate dimensions to maintain reasonable resolution
    # ArcGIS typically limits image size, so we'll use 2000 pixels as max dimension
    lon_diff = abs(bbox[2] - bbox[0])
    lat_diff = abs(bbox[3] - bbox[1])
    aspect_ratio = lon_diff / lat_diff

    if aspect_ratio > 1:
        width = 2000
        height = int(2000 / aspect_ratio)
    else:
        height = 2000
        width = int(2000 * aspect_ratio)

    # Prepare export parameters
    params = {
        "bbox": ",".join(map(str, bbox)),
        "bboxSR": "4326",  # WGS84
        "size": f"{width},{height}",
        "format": "png32",  # Using PNG32 for better quality
        "f": "image",  # Direct image response
        "imageSR": "4326",  # Ensure output is in WGS84
        "transparent": "false",
    }

    if token:
        params["token"] = token

    try:
        # Download image
        response = requests.get(image_service_url, params=params)
        response.raise_for_status()

        # Convert PNG to numpy array
        img = Image.open(io.BytesIO(response.content))
        img_array = np.array(img)

        # If image is RGBA, convert to RGB
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        # Create transform
        transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width, height)

        # Prepare metadata for GeoTIFF
        metadata = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 3,  # RGB
            "dtype": img_array.dtype,
            "crs": "EPSG:4326",
            "transform": transform,
            "compress": "lzw",
            "tiled": True,
            "interleave": "pixel",
        }

        # Write GeoTIFF
        print(f"Saving georeferenced image to {output_path}...")
        with rasterio.open(output_path, "w", **metadata) as dst:
            # Reshape array to (bands, height, width)
            img_array = np.moveaxis(img_array, -1, 0)
            dst.write(img_array)

        # Verify the output
        with rasterio.open(output_path) as src:
            print("\nVerification of output file:")
            print(f"CRS: {src.crs}")
            print(f"Transform: {src.transform}")
            print(f"Bounds: {src.bounds}")
            print(f"Size: {src.width} x {src.height}")

        return output_path

    except Exception as e:
        print(f"Error: {str(e)}")
        return None


# Example usage
if __name__ == "__main__":
    # Example bbox for a small area (adjust coordinates as needed)
    bbox = [-122.51, 37.71, -122.35, 37.83]  # San Francisco example

    output_file = "arcgis_imagery_georef.tiff"

    download_arcgis(bbox=bbox, output_path=output_file)
