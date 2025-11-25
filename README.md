# CS4100 Final Project 

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
GOOGLE_CLOUD_STATIC_MAP_API= Your API key of google cloud
STREET_VIEW_IMG_SRC=directory with street view photos
METADATA_OUT=empty directory for holding image metadata
MAP_TILE_IMG_OUT=empty directory for holding map tile data
CNN_PTH_SRC=path of .pth file that hold model weights
```
Fill all env vars before preceeding

#### Build Image metadata and map tile images for models to use

`python src/data-processors/build_dataset.py`

Make sure `STREET_VIEW_IMG_SRC` is filled with street view photos before running.

#### Train CNN model
`python src/cnn/train_model.py`

#### Evaluate a trained CNN model
`python src/cnn/evaluate_model.py`

Make sure you download `observation_model.pth` from github release before running.