"""
Build labeled training dataset using existing MDP.py
Creates pairs of (map_state, observation) with labels for CNN training
"""

import json
from pathlib import Path
from collections import defaultdict
from decimal import Decimal
import numpy as np
from tqdm import tqdm
from MDP import MDP

DATA_PATH = 'Street_View_Photos/Heading'

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