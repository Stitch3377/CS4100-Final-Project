import exiftool
import os
import json

import numpy as np

from tqdm import tqdm
from PIL import Image

parent_dir = "FILE-PATH-HERE"
target_dir = "FILE-PATH-HERE"
exiftool_dir = "FILE-PATH-HERE"#requires to download exiftool to get exiftoolpy to work

cb = np.array([42.332417, -71.093694])#base coord
cl = np.array([42.342028, -71.094639])#coord sharing edge with cb
cw = np.array([42.333222, -71.083861])#coord sharing edge with cb

l = (cl - cb)
w = (cl - cb)

exif_helper = exiftool.ExifToolHelper(executable=exiftool_dir)

for file in tqdm(os.listdir(parent_dir)):
    path = parent_dir + "/" + file
    metadata = exif_helper.get_metadata(path)

    latRef = metadata[0]['EXIF:GPSLatitudeRef']
    lat    = metadata[0]['EXIF:GPSLatitude']
    lonRef = metadata[0]['EXIF:GPSLongitudeRef']
    lon    = metadata[0]['EXIF:GPSLongitude']
    rot    = metadata[0]['EXIF:GPSImgDirection']

    if latRef == "S":
        lat = -lat
    if lonRef == "W":
        lon = -lon

    c = np.array([lat, lon])
    c = c - cb

    if np.dot(c, l) > np.dot(l, l) or np.dot(c, w) > np.dot(w, w):
        continue #out of bounds

    out_folder = target_dir + f"/{lat}-{lon}-{rot}"

    os.mkdir(out_folder)

    img = Image.open(path)

    img = img.resize((640, 640))

    img.save(out_folder + "/gsv_0.jpg", "JPEG")

    new_metadata = {
        "location": {
           "lat": lat,
            "lon": lon,
        },
        "status": "OK",
        "_file": "gsv_0.jpg",

    }
    with open(out_folder + "/metadata.json", "w") as f:
        json.dump(new_metadata, f)

exif_helper.terminate()