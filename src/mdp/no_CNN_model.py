from random import randint

import numpy as np
from tqdm import tqdm
import os
import json

import matplotlib.pyplot as plt
from src.mdp.MDP import MDP
import src.mdp.HMM as HMM

parent_dir = "C:/Users/arbla/PycharmProjects/Street-View-Downloader/photos"

cb = np.array([42.332417, -71.093694])#base coord
cl = np.array([42.342028, -71.094639])#coord sharing edge with cb
cw = np.array([42.333222, -71.083861])#coord sharing edge with cb


sl = np.linalg.norm(cl-cb)
sw = np.linalg.norm(cw-cb)


def build_mdp(l_segments, w_segments):
    mdp_builder = MDP.MDP_builder(cb, cl, cw, l_segments,  w_segments)
    for folder_name in tqdm(os.listdir(parent_dir)):
        folder_path = os.path.join(parent_dir, folder_name)

        # Check if the item is a directory
        if os.path.isdir(folder_path):
            json_filename = "metadata.json"
            json_path = os.path.join(folder_path, json_filename)

            # Check if the expected JSON file exists
            if os.path.isfile(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if data[0]['status'] == 'ZERO_RESULTS':
                            continue
                        y = data[0]['location']['lat']
                        x = data[0]['location']['lng']
                        mdp_builder.feed_photo_loc(np.array([y, x]))


                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in {json_path}: {e}")
            else:
                print(f"No JSON file named {json_filename} found in {folder_path}")

    return mdp_builder.create()

class dummy_cnn:
    def __init__(self, states):
        self.prob = np.ones((states))

    def get_observations(self, observation):
        return self.prob

mdp = build_mdp(50, 50)
cnn = dummy_cnn(mdp.get_total_states())
plt.imshow(mdp.state_grid)
plt.colorbar()
plt.show()
hmm = HMM.HMM(mdp, cnn, None)
action = None
while action != "q":
    action = input()
    if str.isdigit(action):
        hmm.step(int(action), None)
    print(f"possible states: {np.sum(hmm.belief_state)}")
