# pylint: disable=C0303, C0301
"""
Build labeled training dataset using existing MDP.py
Creates pairs of (map_state, observation) with labels for CNN training
"""

import sys
import json
import random
from pathlib import Path
from collections import defaultdict
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from src.mdp.MDP import MDP
from map import BBox, download_campus_map
import os
from dotenv import load_dotenv
load_dotenv()

DATA_PATH = os.getenv("STREET_VIEW_IMG_SRC")
OUT_PATH = os.getenv("METADATA_OUT") 
MAP_CACHE_PATH = os.getenv("MAP_TILE_IMG_OUT")

class DatasetBuilder:
    """Builds labeled training pairs for observation model CNN"""
    
    def __init__(self, mdp, bbox, api_key, map_zoom=18, map_size=(256, 256)):
        """
        Initialize dataset builder with MDP and map generation parameters
        
        Args:
            mdp: MDP instance with grid already configured
            bbox: BBox object defining campus boundaries
            api_key: Google Maps API key for downloading map tiles
            map_zoom: Zoom level for map tiles
            map_size: Size of map tile images (width, height)
        """
        self.mdp = mdp
        self.bbox = bbox
        self.api_key = api_key
        self.map_zoom = map_zoom
        self.map_size = map_size
        self.num_states = np.sum(mdp.state_grid)

        Path(MAP_CACHE_PATH).mkdir(parents=True, exist_ok=True)

        self.state_map_cache = {}

    def generate_all_map_tiles(self):
        """
        Generate map tiles for ALL states in the MDP grid.
        This is done BEFORE pairing with street view images.
        
        Returns:
            Dictionary mapping state_id -> map_tile_path
        """
        state_map_paths = {}
        state_id = 0

        # Iterate through entire grid
        for y in range(self.mdp.state_grid.shape[0]):
            for x in range(self.mdp.state_grid.shape[1]):
                # Only create tiles for valid states (where state_grid == 1)
                if self.mdp.state_grid[y, x]:
                    try:
                        map_path = self.get_or_create_map_tile(state_id)
                        state_map_paths[state_id] = map_path
                    except Exception as e:
                        print(f"Error generating map for state {state_id} at ({y}, {x}): {e}")
                    state_id += 1

        return state_map_paths
    
    def get_or_create_map_tile(self, state_id):
        """
        Get cached map tile for a state or create new one
        
        Args:
            state_id: State ID
            
        Returns:
            Path to map tile image
        """
        cache_path = Path(MAP_CACHE_PATH) / f"state_{state_id}.png"

        if cache_path.exists():
            return str(cache_path)
        
        # Get the bottom-left corner of the state cell
        center_coord = self.mdp.state_to_coord(state_id, center=False)
        corner_lat, corner_lon = float(center_coord[0]), float(center_coord[1])

        # Calculate the full cell size
        # delta_l is the vector for one cell in the length direction
        # delta_w is the vector for one cell in the width direction
        lat_delta = abs(float(self.mdp.delta_l[0]))
        lon_delta = abs(float(self.mdp.delta_w[1]))
        
        # Create bounding box covering the entire grid cell
        state_bbox = BBox(
            lat_min=corner_lat,
            lon_min=corner_lon,
            lat_max=corner_lat + lat_delta,
            lon_max=corner_lon + lon_delta
        )
        
        print(f"    Corner: ({corner_lat:.6f}, {corner_lon:.6f})")
        print(f"    Deltas: lat={lat_delta:.6f}, lon={lon_delta:.6f}")
        print(f"    Cell size: {lat_delta:.6f} x {lon_delta:.6f}")


        # Download map tile
        map_img = download_campus_map(
            bbox=state_bbox,
            zoom=self.map_zoom,
            size=self.map_size,
            api_key=self.api_key,
            maptype='satellite'
        )

        # Save to cache
        map_img.save(cache_path)
        return str(cache_path)

    def collect_images_by_state(self, image_dir=DATA_PATH):
        """
        Collect all street view images and organize by state using MDP's coord_to_state
        Also generates map tiles for each state
        
        Args:
            image_dir: Root directory containing images
            
        Returns:
            Dictionary mapping state_id -> list of image data
        """
        # Search for JSON metadata files
        image_dir = Path(image_dir)
        metadata_files = list(image_dir.rglob('*.json'))
        # Filter out Zone Identifier files
        metadata_files = [f for f in metadata_files if not f.name.endswith('.Zone.Identifier')]

        state_images = defaultdict(list)
        out_of_bounds = 0
        errors = 0
        no_image_file = 0

        for meta_file in tqdm(metadata_files, desc='Processing images'):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Handle list or dict format
                if isinstance(metadata, list):
                    if len(metadata) == 0:
                        continue
                    metadata = metadata[0]
                
                # Check if image was fetched
                if metadata.get('status') != 'OK':
                    continue

                # Get image coordinates
                lat = metadata['location']['lat']
                lon = metadata['location']['lng']
                coord = np.array([float(lat), float(lon)])

                # Search for corresponding image file
                image_file = None
                possible_names = [
                    meta_file.with_suffix('.jpg'),
                    meta_file.parent / meta_file.stem.replace('metadata', 'gsv_0'),
                    meta_file.parent / 'gsv_0.jpg'
                ]

                for possible_file in possible_names:
                    if possible_file.exists() and not possible_file.name.endswith('.Zone.Identifier'):
                        image_file = possible_file
                        break

                if image_file is None: # if name doesn't match, just pick JPG in file
                    for jpg_file in meta_file.parent.glob('*.jpg'):
                        if not jpg_file.name.endswith('.Zone.Identifier'):
                            image_file = jpg_file
                            break
                        
                if image_file is None or not image_file.exists():
                    no_image_file += 1
                    continue

                # Use MDP to determine state
                try:
                    state_id = self.mdp.coord_to_state(coord)

                    try:
                        heading = float(meta_file.parent.name.split('_')[-1])
                    except (ValueError, IndexError):
                        heading = 0.0

                    state_images[state_id].append({
                        'image_path': str(image_file),
                        'metadata_path': str(meta_file),
                        'lat': float(lat),
                        'lon': float(lon),
                        'heading': heading,
                        'coord': coord,
                        'pano_id': metadata.get('pano_id', ''),
                        'date': metadata.get('date', '')
                    })
                except (IndexError, ValueError):
                    out_of_bounds += 1
            except Exception as e:
                print(f"Error {errors}: {e}")
                errors += 1
                continue

        total_images = sum(len(imgs) for imgs in state_images.values())
        print("Image collection complete:")
        print(f"Total images found: {total_images}")
        print(f"States with images: {len(state_images)}")
        print(f"Out of bounds: {out_of_bounds}")
        print(f"Missing image files: {no_image_file}")
        print(f"Errors: {errors}")

        return dict(state_images)

    def get_state_center_coord(self, state_id):
        """
        Get center coordinate of a state using MDP's state_to_coord
        
        Args:
            state_id: State ID
            
        Returns:
            (lat, lon) tuple
        """
        # Get corner coordinate (relative to cb)
        center_coord = self.mdp.state_to_coord(state_id, center=True)
        return (float(center_coord[0]), float(center_coord[1]))
    
    def create_positive_pairs(self, state_images, state_map_paths):
        """
        Create positive training pairs where street view observation matches map state
        
        Args:
            state_images: Dictionary of state_id -> list of street view images
            state_map_paths: Dictionary of state_id -> map_tile_path
            
        Returns:
            List of positive pairs
        """
        positive_pairs = []

        for state_id, images in tqdm(state_images.items(), desc="Creating positive pairs"):
            # Get map tile
            if state_id not in state_map_paths:
                print(f"Warning: No map tile found for state {state_id}")
                continue
            map_path = state_map_paths[state_id]

            center_lat, center_lon = self.get_state_center_coord(state_id)

            for img_data in images:
                pos_pair = {
                    'map_image_path': map_path,
                    'street_view_path': img_data['image_path'],
                    'state_id': int(state_id),
                    'map_center_lat': center_lat,
                    'map_center_lon': center_lon,
                    'observation_lat': img_data['lat'],
                    'observation_lon': img_data['lon'],
                    'heading': img_data.get('heading', 0.0),
                    'pano_id': img_data.get('pano_id', ''),
                    'date': img_data.get('date', ''),
                    'label': 1,
                    'pair_type': 'positive'
                }
                positive_pairs.append(pos_pair)
    
        return positive_pairs

    def get_neighboring_states(self, state_id, valid_states, radius):
        """
        Get neighboring states within a certain grid distance
        
        Args:
            state_id: Center state ID
            valid_states: Set of valid state IDs
            radius: Grid distance radius for neighbors
            
        Returns:
            List of neighboring state IDs
        """

        # Find grid position of target state
        target_y, target_x = None, None
        states_seen = 0

        for y, row in enumerate(self.mdp.state_grid):
            for x, cell in enumerate(row):
                if cell:
                    if states_seen ==  state_id:
                        target_y, target_x = y, x
                        break
                    states_seen += 1
            if target_y is not None:
                break
        
        if target_y is None:
            return []
        
        # Find neighbors within radius
        neighbors = []
        states_seen = 0

        for y, row in enumerate(self.mdp.state_grid):
            for x, cell in enumerate(row):
                if cell:
                    dist = abs(y - target_y) + abs(x - target_x)
                    if 0 < dist <= radius and states_seen in valid_states:
                        neighbors.append(states_seen)
                    states_seen += 1

        return neighbors

    def create_negative_pairs(self, positive_pairs, state_images, state_map_paths, target_positive_ratio=0.1, hard_negative_ratio=0.5):
        """
        Create negative training pairs where observation doesn't match map state.
        Hard negatives are states nearby to the target that are matched incorrectly.
        Easy negatives are random far away states to the target that are matched incorrectly.
        
        Args:
            positive_pairs: List of positive pairs
            state_images: Dictionary of state_id -> list of images
            state_map_paths: Dictionary of state_id -> map_tile_path
            target_positive_ratio: Target ratio of positive pairs
            hard_negative_ratio: Fraction of negatives from nearby states
            
        Returns:
            List of negative pairs
        """
        num_positives = len(positive_pairs)
        num_negatives_needed = int(num_positives) * (1 - target_positive_ratio) / target_positive_ratio
        negatives_per_positive = num_negatives_needed // num_positives
        valid_states = set(state_images.keys())
        all_states_with_maps = set(state_map_paths.keys())
        negative_pairs = []

        for pos_pair in tqdm(positive_pairs, desc="Creative negative pairs"):
            true_state_id = pos_pair['state_id']
            street_view_path = pos_pair['street_view_path']
            obs_lat = pos_pair['observation_lat']
            obs_lon = pos_pair['observation_lon']
            pano_id = pos_pair['pano_id']
            date = pos_pair['date']
            heading = pos_pair['heading']

            # Get neighboring states for hard negatives
            neighbors = self.get_neighboring_states(true_state_id, valid_states, radius=3)
            far_states = [s for s in all_states_with_maps if s != true_state_id and s not in neighbors]

            # Calculate how many hard vs. easy negatives
            num_hard = int(negatives_per_positive * hard_negative_ratio)
            num_easy = int(negatives_per_positive - num_hard)

            # Sample easy and hard negatives
            hard_samples = random.sample(neighbors, min(num_hard, len(neighbors))) if neighbors else []
            easy_samples = random.sample(far_states, min(num_easy, len(far_states))) if far_states else []

            # Create negative pairs with wrong map tiles
            for neg_state_id in hard_samples + easy_samples:
                # Get map tiles
                if neg_state_id not in state_map_paths:
                    print(f"Warning: No map tile found for state {neg_state_id}")
                    continue
                map_path = state_map_paths[neg_state_id]
                center_lat, center_lon = self.get_state_center_coord(neg_state_id)
                
                neg_pair = {
                    'map_image_path': map_path,
                    'street_view_path': street_view_path,
                    'state_id': int(neg_state_id),
                    'map_center_lat': center_lat,
                    'map_center_lon': center_lon,
                    'observation_lat': obs_lat,
                    'observation_lon': obs_lon,
                    'heading': heading,
                    'pano_id': pano_id,
                    'date': date,
                    'label': 0,
                    'true_state_id': int(true_state_id),
                    'pair_type': 'hard_negative' if neg_state_id in hard_samples else 'easy_negative'
                }
                negative_pairs.append(neg_pair)
        return negative_pairs
    
    def build_dataset(self, image_dir=DATA_PATH, target_positive_ratio=0.1, hard_negative_ratio=0.5, random_seed=42):
        """
        Build complete dataset with positive and negative pairs
        
        Args:
            image_dir: Root directory containing street view images
            target_positive_ratio: Target ratio of positive pairs (0.10 = 10%)
            hard_negative_ratio: Ratio of hard negatives
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (dataset, metadata)
        """
        random.seed(random_seed)
        np.random.seed(random_seed)

        print("\n" + "="*60)
        print("BUILDING DATASET")
        print("="*60)

        # 1. Generate map tiles
        state_map_paths = self.generate_all_map_tiles()

        # 2. Collect images by state
        state_images = self.collect_images_by_state(image_dir)

        if len(state_images) == 0:
            raise ValueError("No images found. Check image directory.")
        
        # 3. Create positive pairs
        positive_pairs = self.create_positive_pairs(state_images, state_map_paths)

        # 4. Create negative pairs
        negative_pairs = self.create_negative_pairs(
            positive_pairs, state_images, state_map_paths,
            target_positive_ratio=target_positive_ratio,
            hard_negative_ratio=hard_negative_ratio
        )
        
        # 5. Combine & shuffle
        dataset = positive_pairs + negative_pairs
        random.shuffle(dataset)

        # Calculate statistics for metadata
        total_pairs = len(dataset)
        actual_positive_ratio = len(positive_pairs) / total_pairs
        num_hard = sum(1 for p in negative_pairs if p['pair_type'] == 'hard_negative')
        num_easy = sum(1 for p in negative_pairs if p['pair_type'] == 'easy_negative')

        metadata = {
            'total_pairs': total_pairs,
            'positive_pairs': len(positive_pairs),
            'negative_pairs': len(negative_pairs),
            'hard_negatives': num_hard,
            'easy_negatives': num_easy,
            'actual_positive_ratio': actual_positive_ratio,
            'states_with_images': len(state_images),
            'total_states': int(self.num_states),
            'random_seed': random_seed,
            'target_positive_ratio': target_positive_ratio,
            'hard_negative_ratio': hard_negative_ratio,
            'map_zoom': self.map_zoom,
            'map_size': self.map_size,
            'states_with_map_tiles': len(state_map_paths),
            'grid_shape': [int(x) for x in self.mdp.state_grid.shape]
        }

        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Total pairs: {total_pairs:,}")
        print(f"  Positive: {len(positive_pairs):,} ({actual_positive_ratio*100:.2f}%)")
        print(f"  Negative: {len(negative_pairs):,} ({(1-actual_positive_ratio)*100:.2f}%)")
        print(f"    - Hard negatives: {num_hard:,}")
        print(f"    - Easy negatives: {num_easy:,}")
        print(f"\nStates with images: {len(state_images)} / {self.num_states}")
        print(f"Map tiles cached: {len(list(Path(MAP_CACHE_PATH).glob('*.png')))}")
        print("\nGrid coverage:")
        print(f"  Total states in grid: {self.num_states}")
        print(f"  States with map tiles: {len(state_map_paths)}")
        print(f"  States with street view images: {len(state_images)}")
        print("="*60)
        
        return dataset, metadata

    def save_dataset(self, dataset, metadata, test_size=0.25):
        """
        Save dataset image paths and metadata as JSON, images are loaded during training.

        Args:
            dataset: List of data pairs
            metadata: Dataset metadata dictionary
            test_size: Fraction of data for the test set
        """
        # Create output directory
        Path(OUT_PATH).mkdir(parents=True, exist_ok=True)

        # Train/test split
        train_data, test_data = train_test_split(
            dataset,
            test_size=test_size,
            random_state=metadata['random_seed'],
            stratify=[d['label'] for d in dataset]
        )

        # Save as JSON files
        train_path = Path(OUT_PATH) / 'train_pairs.json'
        test_path = Path(OUT_PATH) / 'test_pairs.json'
        metadata_path = Path(OUT_PATH) / 'dataset_metadata.json'
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2)
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)

        # Add split info to metadata
        metadata['train_samples'] = len(train_data)
        metadata['test_samples'] = len(test_data)
        metadata['test_size'] = test_size
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

def main():
    """Main function to build dataset"""
    # Define campus boundary
    cb = np.array([42.332417, -71.093694]) # base coord
    cl = np.array([42.342028, -71.094639]) # coord sharing edge with cb
    cw = np.array([42.333222, -71.083861]) # coord sharing edge with cb

    # Create MDP & feed street view locations to populate grid
    builder = MDP.MDP_builder(cb, cl, cw, 25, 25)
    image_dir = Path(DATA_PATH)
    metadata_files = list(image_dir.rglob('*.json'))
    metadata_files = [f for f in metadata_files if not f.name.endswith('.Zone.Identifier')]

    for meta_file in tqdm(metadata_files, desc="Populating MDP grid"):
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            if isinstance(metadata, list) and len(metadata) > 0:
                metadata = metadata[0]
            if metadata.get('status') == 'OK':
                lat = metadata['location']['lat']
                lon = metadata['location']['lng']
                coord = np.array([float(lat), float(lon)])
                builder.feed_photo_loc(coord)
        except Exception as e:
            print(e)
            continue
    mdp = builder.create()

    # Define bounding box
    bbox = BBox(
        lat_min=float(cb[0]),
        lon_min=float(cb[1]),
        lat_max=float(cl[0]),
        lon_max=float(cw[1])
    )

    # Maps API Key
    api_key = os.getenv("GOOGLE_CLOUD_STATIC_MAP_API")

    # Build dataset
    dataset_builder = DatasetBuilder(
        mdp=mdp,
        bbox=bbox,
        api_key=api_key,
        map_zoom=18,
        map_size=(256, 256)
    )
    
    dataset, metadata = dataset_builder.build_dataset(
        image_dir=DATA_PATH,
        target_positive_ratio=0.1,
        hard_negative_ratio=0.5
    )

    # Save dataset
    dataset_builder.save_dataset(dataset, metadata, test_size=0.25)

if __name__ == "__main__":
    main()
