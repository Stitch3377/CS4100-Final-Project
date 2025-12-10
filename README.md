# HuskyVision

## Overview 
Geoguessor for Northeastern's campus. 

See the usage section to run the demo

### Project structure
- `src/data-processors`  
scripts that download or process image data for models to use
- `src/cnn` 
scripts for training/evaluating a cnn model
- `src/mdp`
scripts for building and running the pomdp
- `data/`
resources for raw image data, image metadata, and .pth weights
- `data/ppt`
resources for data that we use in the powerpoint

### Usage 
#### Install Dependencies
`pip install -r requirements.txt`
#### Configure Environment variables
`cp .env.example .env`

```
GOOGLE_CLOUD_STATIC_MAP_API=Your google cloud Street View Static API key
STREET_VIEW_IMG_SRC=directory with street view photos
METADATA_OUT=empty directory for holding image metadata
MAP_TILE_IMG_OUT=empty directory for holding map tile data
CNN_PTH_SRC=path of .pth file that hold model weights
```
Fill all env vars before preceeding. You just need to fill the api key if you are just running the demo.

#### Run Demo
1. Download `best_model.pth' from github release and place it at the project's root repositiory.

2. Run `python src/demo.py`
3. It will prompt you to give a starting location for our agent(that our agent is unaware of). Input a laltitude within [42.332417, 42.342028], and a longtitude within [ -71.093694, -71.094639]
4. The demo script will download a random image of that location from google street view and feed it to the agent. You will see a map, where the red x denotes the agent's location, the black areas are where the agent believes he is not in, and the white areas are where the agent believes he is in. 
5. Close the map. You will be prompted to input a move using an index from [0,7]. The index corresponds to the following map:
```
ACTIONS = np.array([
    [0,1],[1,0],[0,-1],[-1,0],#UP,LEFT,DOWN,RIGHT
    [-1,-1],[-1,1],[1,-1],[1,1],#Diagonals (DL, UL, DR, UR)
])
```
6. You will see the updated map. Repeat and see the power of our agent!

You do not need to do the following items if you just want to see the demo.


#### Build Image metadata and map tile images for models to use

`python src/data-processors/build_dataset.py`

Make sure `STREET_VIEW_IMG_SRC` is filled with street view photos before running.

#### Train CNN model
`python src/cnn/train_model.py`

#### Evaluate a trained CNN model
`python src/cnn/evaluate_model.py`

Make sure you download `observation_model.pth` from github release before running.

#### Run tests
`pytest`