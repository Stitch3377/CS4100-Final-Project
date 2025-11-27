from random import random

import numpy as np
from tqdm import tqdm
import os
import json

import matplotlib.pyplot as plt
from src.mdp.MDP import MDP
import src.mdp.MDP as mdpClass
import src.cnn.api as cnnAPI
import google_streetview.api
import matplotlib.image as mpimg

from dotenv import load_dotenv

import src.mdp.HMM as HMM

load_dotenv()

photos_dir = os.getenv("STREET_VIEW_IMG_SRC")
model_dir = os.getenv("CNN_PTH_SRC")
map_dir = os.getenv("MAP_TILE_IMG_OUT")
api_key = os.getenv("GOOGLE_CLOUD_STATIC_MAP_API")


cb = np.array([42.332417, -71.093694])#base coord
cl = np.array([42.342028, -71.094639])#coord sharing edge with cb
cw = np.array([42.333222, -71.083861])#coord sharing edge with cb

sl = np.linalg.norm(cl-cb)
sw = np.linalg.norm(cw-cb)

l = cl - cb
w = cw - cb

def build_mdp(l_segments, w_segments):
    mdp_builder = MDP.MDP_builder(cb, cl, cw, l_segments,  w_segments)
    for folder_name in tqdm(os.listdir(photos_dir)):
        folder_path = os.path.join(photos_dir, folder_name)

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

class CNN_wrapper:
    def __init__(self, states):
        self.cnn_api = cnnAPI.StreetViewMatcher(model_path=model_dir, map_dir=map_dir)
        self.states = states

    def get_observations(self, photo):
        observations = np.zeros(self.states)
        for state in range(self.states):
            observations[state] = self.cnn_api.get_match_probability(photo, state)
        return observations


def get_photo(latitude, longitude, heading):
    params = [{
        'size': '640x640',
        'location': str(latitude) + ',' + str(longitude),
        'heading': str(heading),
        'pitch': '0',
        'key': api_key
    }]
    path = f"Demo-Path/p{latitude}_{longitude}_{heading}"

    results = google_streetview.api.results(params)
    results.download_links(path)
    return path + "/gsv_0.jpg"

#Controller for the demo

l_segments = 25
w_segments = 25

lat = input("input starting latitude: ")
lng = input("input starting longitude: ")

current_coordinate = np.array([float(lat), float(lng)])

mdp = build_mdp(l_segments, w_segments)
cnn = CNN_wrapper(mdp.get_total_states())
photo_path = get_photo(current_coordinate[0], current_coordinate[1], 360 * random())
hmm = HMM.HMM(mdp, cnn, photo_path)
photo = mpimg.imread(photo_path)

delta_l = l/l_segments
delta_w = w/w_segments

user_input = None

while user_input != "exit":
    grid_coord = current_coordinate - cb

    norm_w = delta_w / np.dot(delta_w, delta_w)
    norm_l = delta_l / np.dot(delta_l, delta_l)

    pr_w = np.dot(grid_coord, norm_w)
    pr_l = np.dot(grid_coord, norm_l)

    print("current coordinate: ", current_coordinate)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(hmm.belief_state_grid(), cmap='gray')
    axs[0].plot(pr_w,pr_l, "rx", markersize=10)
    axs[1].imshow(photo)
    plt.show()

    user_input = input("input action or quit: ")

    if user_input == "quit":
        break
    if not (str.isdigit(user_input) and 0 <= int(user_input) < len(mdpClass.ACTIONS)):
        print("Invalid input. Please try again.")
        continue

    action = np.flip(mdpClass.ACTIONS[int(user_input)]) * np.array([sl/l_segments, sw/w_segments])#make sure order is correct
    new_coordinate = current_coordinate + action
    if mdp.coord_to_state(new_coordinate) is None:
        print("Action failed as next state does not exist. Please try again.")
        continue
    current_coordinate = new_coordinate
    photo_path = get_photo(current_coordinate[0], current_coordinate[1], 360 * random())
    hmm.step(int(user_input), photo_path)
    photo = mpimg.imread(photo_path)

