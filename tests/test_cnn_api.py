import os
import pytest
from dotenv import load_dotenv

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from cnn.api import StreetViewMatcher


load_dotenv()

CHECKPOINT_PATH = os.getenv("CNN_PTH_SRC")
MAP_CACHE_PATH = os.getenv("MAP_TILE_IMG_OUT")

street_view_matcher = StreetViewMatcher(
        model_path=CHECKPOINT_PATH,
        map_dir=MAP_CACHE_PATH
) 

def test_street_view_matcher():
    """
    tests the following two matches from test data
  {
    "map_image_path": "data/maps/state_43.png",
    "street_view_path": "data/Heading/p1801_42.33347129_-71.09045776_251.00172143759187/gsv_0.jpg",
    "state_id": 43,
    "map_center_lat": 42.3336196,
    "map_center_lon": -71.0908386,
    "observation_lat": 42.33372474618429,
    "observation_lon": -71.09073693267229,
    "heading": 251.00172143759187,
    "pano_id": "RFTkT5zRj9s_qxz0UWuVjw",
    "date": "2025-07",
    "label": 1,
    "pair_type": "positive"
  },

   {
    "map_image_path": "data/maps/state_243.png",
    "street_view_path": "data/Heading/p2622_42.33806052_-71.09333760_92.04176147155995/gsv_0.jpg",
    "state_id": 243,
    "map_center_lat": 42.33807188,
    "map_center_lon": -71.0932588,
    "observation_lat": 42.33796031994505,
    "observation_lon": -71.09303446211896,
    "heading": 92.04176147155995,
    "pano_id": "81BMPCvHK6GgCNmmg9-6lw",
    "date": "2025-07",
    "label": 1,
    "pair_type": "positive"
  },
"""

    STREET_VIEW_1 = "data/Heading/p1801_42.33347129_-71.09045776_251.00172143759187/gsv_0.jpg"
    STREET_VIEW_2 = "data/Heading/p2622_42.33806052_-71.09333760_92.04176147155995/gsv_0.jpg"

    # matching with correct cells
    prob_1_correct = street_view_matcher.get_match_probability(STREET_VIEW_1, 43)
    assert prob_1_correct == pytest.approx(0.6845, rel=1e-3)
        
    prob_2_correct = street_view_matcher.get_match_probability(STREET_VIEW_2, 243)
    assert prob_2_correct == pytest.approx(0.5897, rel=1e-3)

    # matching with wrong cells
    prob_1_wrong = street_view_matcher.get_match_probability(STREET_VIEW_1, 100)
    assert prob_1_wrong == pytest.approx(0.4831, rel=1e-3)
        
    prob_2_wrong = street_view_matcher.get_match_probability(STREET_VIEW_2, 200)
    assert prob_2_wrong == pytest.approx(0.5497, rel=1e-3)
    