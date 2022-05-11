from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import numpy as np
import os
from dtaidistance import dtw

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def remove_pt(word):
    filter_ = ["-s_cm", "-s_pt", "-s_qo", "-s_qt", "-s_sq"]
    if any(word.endswith(pt) for pt in filter_):
        index = max(word.find(pt) for pt in filter_)
        return word[:index]

    return word


def mapping():
    with open("ground-truth/transcription.txt", "r") as t:
        # mapping "position -> word" (not the other way around since word may not be unique)
        pos_to_word = {name: remove_pt(word) for (name, word) in [l.strip().split(" ") for l in t]}
        # get all page numbers
        pages = {k.split("-", 1)[0] for k in pos_to_word.keys()}
        # construct a dict like this: {"270": {"270-01-01": 0, "270-01-02": 1, ...}, "271": ...}
        pos_to_index = {p: {
            k: j for (j, k) in enumerate(sorted([
                i for i in pos_to_word.keys() if i.startswith(p)
            ]))} for p in pages}

    def f(word):
        # reverse-search mapping to find all positions of words
        positions = [(page, pos) for page in pos_to_index.keys()
                        for pos in pos_to_index[page].keys()
                        if pos_to_word[pos] == word]

        return [(f"{page}.jpg", pos_to_index[page][pos]) for (page, pos) in positions]

    return f


word_to_index = mapping()


def parse_svg(path):
    with open(path) as f:
        # this is quite hacky, if someone has a better idea...
        return [
            [
                tuple([int(float(j)) for j in i.split(" ") if "." in j])
                    for i in child.attrib["d"].split("L")
                    if "." in i  # if there is a ".", it means that it is a float number and hence a coordinate
            ]
            
            for child in ET.parse(f).getroot()
        ]


def cut_word(img, pol):
    """
    Cut out a polygon shape from a given (PIL) image
    """

    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(pol, fill=255, outline=None)
    black =  Image.new("L", img.size, 255)

    # get corners of minimal rectangle
    maxx = max(x for (x, _) in pol)
    maxy = max(y for (_, y) in pol)
    minx = min(x for (x, _) in pol)
    miny = min(y for (_, y) in pol)

    return Image.composite(img, black, mask).crop((minx, miny, maxx, maxy))


def crop_images(imgs):
    locations_path = "ground-truth/locations"
    polygons = {f: parse_svg(f"{locations_path}/{f}") for f in os.listdir(locations_path)}

    # cut out every word for every image
    return {
        name: [
            cut_word(Image.fromarray(img.astype('uint8')*255), pol)
                for pol in polygons[f"{name.replace('jpg', 'svg')}"]
        ] for (name, img) in imgs.items()
    }


def get_dist(a, b):
    x = np.array(a.resize((100, a.size[1])), dtype=np.double)
    y = np.array(b.resize((100, b.size[1])), dtype=np.double)
    print(x.flatten().shape)
    print(y.flatten().shape)

    distance , _ = fastdtw(x, y, dist=euclidean)
    # distance = dtw.distance_fast(x.flatten(), y.flatten(), use_pruning=True)
    return distance


def dist_mat(a, b):
    return np.array([[get_dist(i, j) for j in b] for i in a])


def contours(img):
    idx = img.nonzero()
    nonzeros = [(r, c.nonzero()[-1]) for r, c in enumerate(1-img.T) if len(c.nonzero()[0]) > 0]
    cols = [i for (i, j) in nonzeros]
    rows_upper = [np.max(j) for (i, j) in nonzeros]
    rows_lower = [np.min(j) for (i, j) in nonzeros]
    z = (1 - np.zeros(img.shape)).T
    z[cols, rows_upper] = -1
    z[cols, rows_lower] = -1

    return z.T