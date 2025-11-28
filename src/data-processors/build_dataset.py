import numpy as np
import cv2
import os
import math
import re

from dotenv import load_dotenv
load_dotenv()

# Fallback paths
MAP_CACHE_PATH = os.getenv("MAP_TILE_IMG_OUT", "data/maps")
MAP_TRANSFORMED_PATH = os.getenv("MAP_TILE_TRANSFORMED_OUT", "data/transformed")

class PolarTransformer:
    def __init__(self, output_height=112, full_width=616):
        self.height = output_height
        self.full_width = full_width
        self.crop_width = int(full_width / 4) 

    def sample_bilinear(self, signal, rx, ry):
        """Standard bilinear sampling."""
        h, w, _ = signal.shape
        
        # Clip coordinates to be inside the image
        rx = np.clip(rx, 0, h - 1)
        ry = np.clip(ry, 0, w - 1)

        ix0 = rx.astype(int)
        iy0 = ry.astype(int)
        ix1 = np.clip(ix0 + 1, 0, h - 1)
        iy1 = np.clip(iy0 + 1, 0, w - 1)
        
        signal_00 = signal[ix0, iy0]
        signal_10 = signal[ix1, iy0]
        signal_01 = signal[ix0, iy1]
        signal_11 = signal[ix1, iy1]
        
        na = np.newaxis
        fx1 = (ix1 - rx)[..., na] * signal_00 + (rx - ix0)[..., na] * signal_10
        fx2 = (ix1 - rx)[..., na] * signal_01 + (rx - ix0)[..., na] * signal_11
        
        return (iy1 - ry)[..., na] * fx1 + (ry - iy0)[..., na] * fx2

    def get_pixel_offset(self, center_latlon, target_latlon, meters_per_pixel):
        """Calculates pixel (row, col) offset of target relative to center."""
        lat_c, lon_c = center_latlon
        lat_t, lon_t = target_latlon
        
        R = 6378137 
        d_lat = (lat_t - lat_c) * (np.pi / 180) * R
        avg_lat = (lat_t + lat_c) / 2 * (np.pi / 180)
        d_lon = (lon_t - lon_c) * (np.pi / 180) * R * np.cos(avg_lat)

        dx = d_lon / meters_per_pixel # East/West
        dy = -d_lat / meters_per_pixel # North/South (Negative because Row 0 is Top)
        return dy, dx

    def stitch_neighborhood(self, center_entry, neighbor_entries, meters_per_px):
        """
        Stitches the center tile and its neighbors into one large Super-Tile.
        Returns the Super-Tile image and the (row, col) of the center tile's origin within it.
        """
        # Load Center Image
        center_path = os.path.join(MAP_CACHE_PATH, os.path.basename(center_entry['map_image_path']))
        if not os.path.exists(center_path): return None, (0,0)
        center_img = cv2.imread(center_path)
        if center_img is None: return None, (0,0)

        h, w, c = center_img.shape
        
        # Create a large canvas (3x3 grid size approximation)
        # We start with a canvas 3x size of single tile
        canvas_h, canvas_w = h * 3, w * 3
        canvas = np.zeros((canvas_h, canvas_w, c), dtype=np.uint8)
        
        # Place center tile in the middle of canvas
        start_y = h
        start_x = w
        canvas[start_y:start_y+h, start_x:start_x+w] = center_img
        
        # Origin of the center tile within the canvas
        origin_offset = (start_y, start_x)
        
        center_coords = (center_entry['map_center_lat'], center_entry['map_center_lon'])

        # Place Neighbors
        for nb in neighbor_entries:
            nb_path = os.path.join(MAP_CACHE_PATH, os.path.basename(nb['map_image_path']))
            if not os.path.exists(nb_path): continue
            nb_img = cv2.imread(nb_path)
            if nb_img is None: continue
            
            # Calculate where this neighbor goes relative to center
            nb_coords = (nb['map_center_lat'], nb['map_center_lon'])
            dy, dx = self.get_pixel_offset(center_coords, nb_coords, meters_per_px)
            
            # Calculate placement on canvas
            # We round to nearest int to place the image
            place_y = int(start_y + dy)
            place_x = int(start_x + dx)
            
            # Determine overlapping regions to blend/overwrite
            # Simple overwrite logic (Painter's Algorithm)
            # Clip to canvas bounds
            y1, y2 = max(0, place_y), min(canvas_h, place_y + h)
            x1, x2 = max(0, place_x), min(canvas_w, place_x + w)
            
            # Source indices (if neighbor is partially off canvas)
            sy1 = max(0, -place_y)
            sy2 = sy1 + (y2 - y1)
            sx1 = max(0, -place_x)
            sx2 = sx1 + (x2 - x1)
            
            if y2 > y1 and x2 > x1:
                canvas[y1:y2, x1:x2] = nb_img[sy1:sy2, sx1:sx2]
                
        # Re-paste center tile on top to ensure the primary tile is pristine
        canvas[start_y:start_y+h, start_x:start_x+w] = center_img
        
        return canvas, origin_offset

    def process_with_neighbors(self, center_entry, neighbor_entries):
        """
        Main function to process a state using its neighbors for context.
        """
        # 1. Setup metadata
        map_filename = os.path.basename(center_entry["map_image_path"])
        out_filename = f"{center_entry['state_id']}_{center_entry['pano_id']}.png"
        output_path = os.path.join(MAP_TRANSFORMED_PATH, out_filename)
        
        street_path = center_entry.get("street_view_path", "")
        heading = 0.0
        match = re.search(r'_([\d\.]+)/[^/]+$', street_path)
        if match:
            try: heading = float(match.group(1))
            except ValueError: pass
            
        lat = center_entry["map_center_lat"]
        # Approx meters per pixel at zoom 18 (Standard Google Maps)
        meters_per_px = 156543.03 * np.cos(np.deg2rad(lat)) / (2**18) 
        # Note: If your tiles are zoom 19, change 18 to 19 above!

        # 2. Create Super-Tile (Stitch)
        super_map, origin_offset = self.stitch_neighborhood(center_entry, neighbor_entries, meters_per_px)
        if super_map is None: return

        # 3. Locate Camera in Super-Tile
        # First, find cam loc relative to center tile center
        map_center = (center_entry["map_center_lat"], center_entry["map_center_lon"])
        obs_coords = (center_entry["observation_lat"], center_entry["observation_lon"])
        
        dy_rel, dx_rel = self.get_pixel_offset(map_center, obs_coords, meters_per_px)
        
        # Center of the "Center Tile" is at h/2, w/2 relative to its top-left
        # And its top-left is at 'origin_offset' in the super map
        h_tile, w_tile = 256, 256 # Assuming 256x256 tiles
        
        tile_center_y = origin_offset[0] + h_tile / 2.0
        tile_center_x = origin_offset[1] + w_tile / 2.0
        
        cam_row_super = tile_center_y + dy_rel
        cam_col_super = tile_center_x + dx_rel

        # 4. Transform
        self.transform_and_crop(
            signal=super_map,
            output_path=output_path,
            center_angle_deg=heading,
            camera_location=(cam_row_super, cam_col_super),
            do_crop=True
        )
        print(f"Processed ID {center_entry['state_id']} using neighbors.")

    def transform_and_crop(self, signal, output_path, center_angle_deg, 
                          camera_location, do_crop=True):
        
        cam_row, cam_col = camera_location
        
        # --- GRID GEN ---
        i = np.arange(0, self.height)
        j = np.arange(0, self.full_width)
        jj, ii = np.meshgrid(j, i)

        theta = 2 * np.pi * jj / self.full_width
        
        # Radius scale should match original tile scale
        # SAFA uses S/2 where S=original image size (256). 
        # Even though our super-map is huge, the "zoom" should be relative to 256.
        max_radius = 256.0 / 2.0 
        radius = max_radius * (self.height - 1 - ii) / self.height

        x_src = cam_row - radius * np.cos(theta) 
        y_src = cam_col + radius * np.sin(theta)

        # --- SAMPLING ---
        full_polar_image = self.sample_bilinear(signal, x_src, y_src)

        # --- CROP ---
        if do_crop:
            center_pixel = (center_angle_deg / 360.0) * self.full_width
            start_pixel = int(center_pixel - (self.crop_width / 2))
            
            col_indices = np.arange(start_pixel, start_pixel + self.crop_width) % self.full_width
            col_indices = col_indices.astype(int)
            final_image = full_polar_image[:, col_indices, :]
        else:
            final_image = full_polar_image

        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
        final_image = np.clip(final_image, 0, 255).astype(np.uint8)
        cv2.imwrite(output_path, final_image)

# --- Usage Example ---
if __name__ == "__main__":
    transformer = PolarTransformer()
    
    # 1. The main data entry (The one causing issues)
    main_entry = {
        "map_image_path": "data/maps/state_395.png",
        "street_view_path": "data/Heading/p1999_42.34155829_-71.08654624_123.89/gsv_0.jpg",
        "state_id": 395,
        "pano_id": "test_stitch",
        "map_center_lat": 42.34131036,
        "map_center_lon": -71.08683696,
        "observation_lat": 42.34151435, # North East of center
        "observation_lon": -71.08655793,
    }
    
    # 2. You would fetch these from your dataset_builder.get_neighboring_states()
    # This is mock data for the neighbor to the North-East
    neighbor_entry = {
        "map_image_path": "data/maps/state_396.png", # Hypothetical neighbor
        "map_center_lat": 42.34131036 + 0.001, # Roughly north
        "map_center_lon": -71.08683696 + 0.001, # Roughly east
        "state_id": 396
    }
    
    neighbors = [neighbor_entry] # Add all 8 neighbors here

    transformer.process_with_neighbors(main_entry, neighbors)