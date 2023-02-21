import numpy as np
from scipy.spatial import distance_matrix

locations = np.zeros((192, 2))
decimation = 16
fs = 50e6 / 16 / decimation
dist_per_cycle = 343_000 / fs

for idx_r, r in enumerate([34.5, 74.5, 124.5, 184, 258.315, 349.5, 461.5, 599.5]):
    for idx_a, a in enumerate(range(0, 360, 15)):
        a_rad = np.pi * a / 180
        locations[idx_a * 8 + idx_r] = (np.sin(a_rad) * r, np.cos(a_rad) * r)

max_offset = np.ceil((distance_matrix(locations, locations) * 1.1 + 10) / dist_per_cycle).astype(np.int32)
bad_mic = [7, 14, 15, 20, 21, 52, 53, 108, 132, 140, 141, 143, 151, 183]


def clean_bad_mics(x):
    x[:, bad_mic] = 0
    x[bad_mic] = 0
    return x


def clean_bad_mics2(x):
    x[bad_mic] = 0
    return x
