import random
import google_streetview.api
from decimal import Decimal, getcontext
from tqdm import tqdm

getcontext().prec = 10

api_key = 'Key Goes Here'

location_samples = 500 #number of photos desired (2 api calls per photo)
last_sample = 0 #index of the last sample taken

cb = (Decimal(42.332417), Decimal(-71.093694))#base coord
cl = (Decimal(42.342028), Decimal(-71.094639))#coord sharing edge with cb
cw = (Decimal(42.333222), Decimal(-71.083861))#coord sharing edge with cb

#calculates vectors for length and width
l = (cl[0] - cb[0], cl[1] - cb[1])
w = (cw[0] - cb[0], cw[1] - cb[1])

for i in tqdm(range(location_samples)):
    #gets random scalars
    rl = Decimal(random.random())
    rw = Decimal(random.random())

    #scalar to length/width making a random point in the grid
    sl = (rl * l[0], rl * l[1])
    sw = (rw * w[0], rw * w[1])

    pos = (sl[0] + sw[0] + cb[0], sl[1] + sw[1] + cb[1])#re-adds base vector
    rot = 360 * random.random()

    #print(f"fetching pos: ({str(pos[0])}, {str(pos[1])}) with rotation {rot}")

    params = [{
        'size': '640x640',
        'location': str(pos[0])+','+str(pos[1]),
        'heading': str(rot),
        'pitch': '0',
        'key': api_key
    }]

    results = google_streetview.api.results(params)
    results.download_links(f"Heading\p{last_sample + i}_{pos[0]}_{pos[1]}_{rot}")

