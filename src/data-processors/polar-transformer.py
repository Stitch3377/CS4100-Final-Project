import numpy as np
import cv2
import os
import math
import re

from dotenv import load_dotenv
load_dotenv()

# Fallback paths if .env is missing
MAP_CACHE_PATH = os.getenv("MAP_TILE_IMG_OUT", "data/maps")
MAP_TRANSFORMED_PATH = os.getenv("MAP_TILE_TRANSFORMED_OUT", "data/transformed")

class PolarTransformer:
    def __init__(self, output_height=112, full_width=616):
        self.height = output_height
        self.full_width = full_width
        self.crop_width = int(full_width / 4) 

    def sample_within_bounds(self, signal, x, y, bounds):
        xmin, xmax, ymin, ymax = bounds
        idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)
        sample = np.zeros((x.shape[0], x.shape[1], signal.shape[-1]))
        sample[idxs, :] = signal[x[idxs], y[idxs], :]
        return sample

    def sample_bilinear(self, signal, rx, ry):
        signal_dim_x = signal.shape[0]
        signal_dim_y = signal.shape[1]
        
        ix0 = rx.astype(int)
        iy0 = ry.astype(int)
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        
        bounds = (0, signal_dim_x, 0, signal_dim_y)
        
        signal_00 = self.sample_within_bounds(signal, ix0, iy0, bounds) 
        signal_10 = self.sample_within_bounds(signal, ix1, iy0, bounds) 
        signal_01 = self.sample_within_bounds(signal, ix0, iy1, bounds) 
        signal_11 = self.sample_within_bounds(signal, ix1, iy1, bounds) 
        
        na = np.newaxis
        fx1 = (ix1 - rx)[..., na] * signal_00 + (rx - ix0)[..., na] * signal_10
        fx2 = (ix1 - rx)[..., na] * signal_01 + (rx - ix0)[..., na] * signal_11
        
        return (iy1 - ry)[..., na] * fx1 + (ry - iy0)[..., na] * fx2

    def get_pixel_coords(self, map_center, obs_coords, image_shape, meters_per_pixel):
        lat_center, lon_center = map_center
        lat_obs, lon_obs = obs_coords
        S_rows, S_cols = image_shape[:2]

        R = 6378137 # Earth Radius
        
        # Calculate distance in meters
        d_lat = (lat_obs - lat_center) * (np.pi / 180) * R
        avg_lat = (lat_obs + lat_center) / 2 * (np.pi / 180)
        d_lon = (lon_obs - lon_center) * (np.pi / 180) * R * np.cos(avg_lat)

        # Convert to pixels (North is Negative Row, East is Positive Col)
        dx_pixels = d_lon / meters_per_pixel
        dy_pixels = d_lat / meters_per_pixel 

        center_row = S_rows / 2.0
        center_col = S_cols / 2.0

        obs_row = center_row - dy_pixels 
        obs_col = center_col + dx_pixels

        return obs_row, obs_col

    def process_data_entry(self, data_entry, do_crop=True):
        map_filename = os.path.basename(data_entry["map_image_path"])
        map_path = os.path.join(MAP_CACHE_PATH, map_filename)
        
        # Unique output filename
        out_filename = f"{data_entry['state_id']}_{data_entry['pano_id']}.png"
        output_path = os.path.join(MAP_TRANSFORMED_PATH, out_filename)

        # Extract Heading
        street_path = data_entry.get("street_view_path", "")
        heading = 0.0
        match = re.search(r'_([\d\.]+)/[^/]+$', street_path)
        if match:
            try:
                heading = float(match.group(1))
            except ValueError: pass
        
        # Calculate Scale
        lat = data_entry["map_center_lat"]
        meters_per_px = 156543.03 * np.cos(np.deg2rad(lat)) / (2**19)

        if not os.path.exists(map_path):
            print(f"Skipping: {map_path} not found.")
            return

        temp_img = cv2.imread(map_path)
        if temp_img is None:
            return

        map_center = (data_entry["map_center_lat"], data_entry["map_center_lon"])
        obs_coords = (data_entry["observation_lat"], data_entry["observation_lon"])
        cam_loc = self.get_pixel_coords(map_center, obs_coords, temp_img.shape, meters_per_px)

        self.transform_and_crop(
            input_path=map_path,
            output_path=output_path,
            center_angle_deg=heading,
            camera_location=cam_loc,
            do_crop=do_crop
        )
        # print(f"Processed ID {data_entry['state_id']}: Cam at {cam_loc}, Heading {heading:.1f}")

    def transform_and_crop(self, input_path, output_path, center_angle_deg, 
                          camera_location=None, do_crop=True):
        if not os.path.exists(input_path):
            return

        signal = cv2.imread(input_path)
        if signal is None:
            return
            
        S_rows, S_cols, _ = signal.shape

        if camera_location is None:
            cam_row, cam_col = S_rows / 2.0, S_cols / 2.0
        else:
            cam_row, cam_col = camera_location

        # --- COORDINATE GENERATION ---
        i = np.arange(0, self.height)
        j = np.arange(0, self.full_width)
        jj, ii = np.meshgrid(j, i)

        # 1. Angle (theta)
        # Map j (0 to width) to (0 to 2pi). 
        # We align 0 radians to NORTH (Up) to match Compass Heading.
        # This simplifies the crop logic significantly.
        theta = 2 * np.pi * jj / self.full_width

        # 2. Radius
        # Max radius = half the image height (standard SAFA scale)
        max_radius = S_rows / 2.0 
        radius = max_radius * (self.height - 1 - ii) / self.height

        # 3. Polar to Cartesian (Compass Corrected)
        # Standard Trig: 0=East (Right), 90=North (Up).
        # Compass: 0=North (Up), 90=East (Right).
        # Conversion: Compass 0 -> Trig 90. Compass 90 -> Trig 0.
        # Formula: Row (y) = -Radius * cos(compass_theta)
        #          Col (x) =  Radius * sin(compass_theta)
        
        # Note: 'theta' here is our 0..2pi sweep. We treat it as Compass Angle (0 is North).
        # Row decreases (goes Up) as we move North (cos(0)=1).
        x_src = cam_row - radius * np.cos(theta) 
        # Col increases (goes Right) as we move East (sin(pi/2)=1).
        y_src = cam_col + radius * np.sin(theta)

        # --- SAMPLING ---
        full_polar_image = self.sample_bilinear(signal, x_src, y_src)

        # --- CROP ---
        if do_crop:
            # Since theta=0 is North, and center_angle_deg is Compass Heading,
            # we can map degrees directly to pixels.
            center_pixel = (center_angle_deg / 360.0) * self.full_width
            start_pixel = int(center_pixel - (self.crop_width / 2))
            
            col_indices = np.arange(start_pixel, start_pixel + self.crop_width) % self.full_width
            col_indices = col_indices.astype(int)
            final_image = full_polar_image[:, col_indices, :]
        else:
            final_image = full_polar_image

        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # --- CRITICAL FIX: Convert Float64 to Uint8 ---
        # OpenCV needs uint8 (0-255) to save correctly.
        final_image = np.clip(final_image, 0, 255).astype(np.uint8)
        
        cv2.imwrite(output_path, final_image)

# --- Usage Example ---
if __name__ == "__main__":
    transformer = PolarTransformer()
    
    # Example Data
    data_sample = {
    "map_image_path": "data/maps/state_371.png",
    "street_view_path": "data/Heading/p2125_42.34078129_-71.08788137_315.10354884302785/gsv_0.jpg",
    "state_id": 371,
    "map_center_lat": 42.34082932,
    "map_center_lon": -71.08797912,
    "observation_lat": 42.34081807117491,
    "observation_lon": -71.08795946067673,
    "heading": 315.10354884302785,
    "pano_id": "eRHW0gk17qu1LhinnoZPuQ",
    "date": "2024-08",
    "label": 1,
    "pair_type": "positive"
  }

    transformer.process_data_entry(data_sample)