import random
import google_streetview.api
from decimal import Decimal, getcontext
from tqdm import tqdm

getcontext().prec = 10

api_key = 'Key Goes Here'

location_samples = 500 #number of photos desired (2 api calls per photo)
last_sample = 2303 #index of the last sample taken

c1 = (Decimal(42.332417), Decimal(-71.093694))
cl = (Decimal(42.342028), Decimal(-71.094639))
cw = (Decimal(42.333222), Decimal(-71.083861))

l = (cl[0] - c1[0],cl[1] - c1[1])
w = (cw[0] - c1[0],cw[1] - c1[1])

for i in tqdm(range(location_samples)):
    rl = Decimal(random.random())
    rw = Decimal(random.random())

    sl = (rl * l[0], rl * l[1])

    sw = (rw * w[0], rw * w[1])

    pos = (sl[0] + sw[0] + c1[0], sl[1] + sw[1] + c1[1])
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

