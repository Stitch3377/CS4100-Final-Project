from dataclasses import dataclass
from typing import Tuple
import io
import requests
from PIL import Image

@dataclass
class BBox:
    """Geographic bounding box for campus."""
    lat_min: float  # south
    lon_min: float  # west
    lat_max: float  # north
    lon_max: float  # east


def download_campus_map(
    bbox: BBox,
    zoom: int,
    size: Tuple[int, int] = (1024, 1024),
    api_key: str = "",
    maptype: str = "satellite",
) -> Image.Image:
    """
    Download a top-down map image covering the campus using Google Static Maps.

    bbox: bounding box of the campus (lat/lon)
    zoom: Google Maps zoom level (you'll tune this so the campus fits nicely)
    size: (width_px, height_px) of the image
    api_key: your Google Maps API key
    maptype: 'satellite', 'roadmap', etc.

    Returns: a PIL.Image object.
    """
    if not api_key:
        raise ValueError("You must pass a Google Maps API key.")

    # Approximate center of the bounding box
    lat_center = (bbox.lat_min + bbox.lat_max) / 2.0
    lon_center = (bbox.lon_min + bbox.lon_max) / 2.0

    width_px, height_px = size

    # Build Google Static Maps URL
    url = (
        "https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat_center},{lon_center}"
        f"&zoom={zoom}"
        f"&size={width_px}x{height_px}"
        f"&maptype={maptype}"
        f"&key={api_key}"
    )

    resp = requests.get(url)
    resp.raise_for_status()

    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return img

# USAGE

# bbox = BBox(
#     lat_min=42.335,  # south of NU
#     lon_min=-71.095, # west
#     lat_max=42.345,  # north
#     lon_max=-71.082  # east
# )

# api_key = "YOUR_API_KEY_HERE"
# campus_img = download_campus_map(bbox, zoom=17, size=(2048, 2048), api_key=api_key)
# campus_img.save("nu_campus_satellite.png")