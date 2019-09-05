import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rgb2gray (rgb):
    # check if it is already greyscale
    if rgb.ndim < 3:
        return rgb
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def invert (gray):
    return np.max(gray) - gray

def remove_noise (gray):
    avgs = np.mean(gray, 0)
    gray = np.clip(gray - avgs, 0, None)
    return gray

def smooth (gray):
    m = np.size(gray, 0)
    n = np.size(gray, 1)
    res = np.zeros([m, n])
    di = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    dj = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
    for i in range(m):
        for j in range(n):
            for k in range(9):
                ni = i + di[k]
                nj = j + dj[k]
                if ni < 0 or ni >= m or nj < 0 or nj >= n:
                    continue
                res[i, j] += gray[ni, nj]
    return res / 9

def threshold (gray, eps):
    thr = (np.min(gray) + np.max(gray)) / 2
    mu1 = np.nanmean(np.where(gray >= thr, gray, np.nan))
    mu2 = np.nanmean(np.where(gray < thr, gray, np.nan))
    newthr = (mu1 + mu2) / 2
    while np.abs(newthr - thr) >= eps:
        thr = newthr
        mu1 = np.nanmean(np.where(gray >= thr, gray, np.nan))
        mu2 = np.nanmean(np.where(gray < thr, gray, np.nan))
        newthr = (mu1 + mu2) / 2
    return thr
    
def gray2bin (gray, eps = 0.01):
    thr = threshold(gray, eps)
    return np.where(gray >= thr, 1, 0)

def preprocess (img):
    gray = smooth(remove_noise(invert(rgb2gray(img))))
    binim = gray2bin(gray)
    return binim