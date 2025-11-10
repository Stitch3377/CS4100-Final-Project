"""
Build labeled training dataset using existing MDP.py
Creates pairs of (map_state, observation) with labels for CNN training
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from decimal import Decimal
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from MDP import MDP

DATA_PATH = 'Street_View_Photos/Heading'
OUT_PATH = 'data'

class DatasetBuilder:
    """Builds labeled training pairs for observation model CNN"""
    
    def __init__(self, mdp):
        """
        Initialize dataset builder with existing MDP
        
        Args:
            mdp: MDP instance with grid already configured
        """
        self.mdp = mdp
        self.num_states = np.sum(mdp.state_grid)

    def collect_images_by_state(self, image_dir=DATA_PATH):
        """
        Collect all images and organize by state using MDP's coord_to_state
        Searches recursively through subdirectories
        
        Args:
            image_dir: Root directory containing images (can be in subdirectories)
            
        Returns:
            Dictionary mapping state_id -> list of image data
        """
        image_dir = Path(image_dir)

        # Search for JSON metadata files
        metadata_files = list(image_dir.rglob('*.json'))
        metadata_files = [f for f in metadata_files if not f.name.endswith('.Zone.Identifier')] # filter out Zone Identifier files

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
                coord = np.array([Decimal(str(lat)), Decimal(str(lon))])

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

                if image_file is None: # if name doesn't match, just pick whatever JPG is in the file
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

                    state_images[state_id].append({
                        'image_path': str(image_file),
                        'metadata_path': str(meta_file),
                        'lat': float(lat),
                        'lon': float(lon),
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
        coord = self.mdp.state_to_coord(state_id)
        return (float(coord[0]), float(coord[1]))
    
    def create_positive_pairs(self, state_images):
        """
        Create positive training pairs where observation matches map state
        
        Args:
            state_images: Dictionary of state_id -> list of images
            
        Returns:
            List of positive pairs
        """
        positive_pairs = []

        for state_id, images in tqdm(state_images.items(), desc="Positive pairs"):
            center_lat, center_lon = self.get_state_center_coord(state_id)

            for img_data in images:
                pair = {
                    'map_state': {
                        'state_id': int(state_id),
                        'center_lat': center_lat,
                        'center-lon': center_lon
                    },
                    'observation': {
                        'image_path': img_data['image_path'],
                        'true_lat': img_data['lat'],
                        'true_lon': img_data['lon'],
                        'pano_id': img_data.get('pano_id', ''),
                        'date': img_data.get('date', '')
                    },
                    'label': 1,
                    'true_state_id': int(state_id),
                    'claimed_state_id': int(state_id),
                    'pair_type': 'positive'
                }
                positive_pairs.append(pair)
    
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

    def create_negative_pairs(self, positive_pairs, state_images, target_positive_ratio=0.1, hard_negative_ratio=0.5):
        """
        Create negative training pairs where observation doesn't match map state.
        Hard negatives are states nearby to the target that are matched incorrectly.
        Easy negatives are random far away states to the target that are matched incorrectly.
        
        Args:
            positive_pairs: List of positive pairs
            state_images: Dictionary of state_id -> list of images
            target_positive_ratio: Target ratio of positive pairs
            hard_negative_ratio: Fraction of negatives from nearby states
            
        Returns:
            List of negative pairs
        """
        num_positives = len(positive_pairs)
        num_negatives_needed = int(num_positives) * (1 - target_positive_ratio) / target_positive_ratio
        negatives_per_positive = num_negatives_needed // num_positives
        valid_states = set(state_images.keys())

        negative_pairs = []

        for pos_pair in tqdm(positive_pairs, desc="Negative pairs"):
            true_state_id = pos_pair['true_state_id']
            observation = pos_pair['observation']

            # Get neighboring states for hard negatives
            neighbors = self.get_neighboring_states(true_state_id, valid_states, radius=3)
            far_states = [s for s in valid_states if s != true_state_id and s not in neighbors]

            # Calculate how many hard vs. easy negatives
            num_hard = int(negatives_per_positive * hard_negative_ratio)
            num_easy = negatives_per_positive - num_hard

            # Sample hard negatives
            if neighbors and num_hard > 0:
                hard_samples = random.sample(neighbors, min(num_hard, len(neighbors)))
            else:
                hard_samples = []
            
            # Sample easy negatives
            if far_states and num_easy > 0:
                easy_samples = random.sample(far_states, min(num_easy, len(far_states)))
            else:
                easy_samples = []

            # Create negative pairs
            for neg_state_id in hard_samples + easy_samples:
                center_lat, center_lon = self.get_state_center_coord(neg_state_id)

                neg_pair = {
                    'map_state': {
                        'state_id': int(neg_state_id),
                        'center_lat': center_lat,
                        'center_lon': center_lon
                    },
                    'observation': {
                        'image_path': observation['image_path'],
                        'true_lat': observation['true_lat'],
                        'true_lon': observation['true_lon'],
                        'pano_id': observation.get('pano_id', ''),
                        'date': observation.get('date', '')
                    },
                    'label': 0,
                    'true_state_id': int(true_state_id),
                    'claimed_state_id': int(neg_state_id),
                    'pair_type': 'hard_negative' if neg_state_id in hard_samples else 'easy_negative'
                }
                negative_pairs.append(neg_pair)
        return negative_pairs
    
    def build_dataset(self, image_dir=DATA_PATH, target_positive_ratio=0.1, hard_negative_ratio=0.5, random_seed=42):
        """
        Build complete dataset with positive and negative pairs
        
        Args:
            image_dir: Root directory containing images (searches recursively)
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

        # 1. Collect images by state
        state_images = self.collect_images_by_state(image_dir)

        if len(state_images) == 0:
            raise ValueError("No images found. Check image directory.")
        
        # 2. Create positive pairs
        positive_pairs = self.create_positive_pairs(state_images)

        # 3. Create negative pairs
        negative_pairs = self.create_negative_pairs(
            positive_pairs, state_images,
            target_positive_ratio=target_positive_ratio,
            hard_negative_ratio=hard_negative_ratio
        )
        
        # 4. Combine & shuffle
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
            'total_states': self.num_states,
            'random_seed': random_seed,
            'target_positive_ratio': target_positive_ratio,
            'hard_negative_ratio': hard_negative_ratio
        }

        print(f"\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Total pairs: {total_pairs:,}")
        print(f"  Positive: {len(positive_pairs):,} ({actual_positive_ratio*100:.2f}%)")
        print(f"  Negative: {len(negative_pairs):,} ({(1-actual_positive_ratio)*100:.2f}%)")
        print(f"    - Hard negatives: {num_hard:,}")
        print(f"    - Easy negatives: {num_easy:,}")
        print(f"\nStates with images: {len(state_images)} / {self.num_states}")
        print("="*60)
        
        return dataset, metadata

    