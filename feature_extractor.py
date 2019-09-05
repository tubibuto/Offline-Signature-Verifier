import numpy as np
import scipy.ndimage.interpolation as ip

def ro (binim, angle):
    transim = ip.rotate(binim, angle)
    ph = np.sum(transim, 1)
    return np.max(ph) / np.count_nonzero(ph)

def baseline_slant_angle (binim, start = -90, end = 90, step = 1):
    angle = start
    maxv = 0
    maxang = start
    while angle < end:
        roval = ro(binim, angle)
        if roval > maxv:
            maxv = roval
            maxang = angle
        angle += step
    return maxang

def bounding_box (binim):
    m = np.size(binim, 0)
    n = np.size(binim, 1)
    mini, maxi, minj, maxj = m, 0, n, 0
    for i in range(m):
        for j in range(n):
            if binim[i, j] == 0:
                continue
            mini = np.min([mini, i])
            maxi = np.max([maxi, i])
            minj = np.min([minj, j])
            maxj = np.max([maxj, j])
    return mini, minj, maxi, maxj

def aspect_ratio (binim):
    mini, minj, maxi, maxj = bounding_box(binim)
    return (maxj - minj) / (maxi - mini)

def normalized_area (binim):
    mini, minj, maxi, maxj = bounding_box(binim)
    p = (maxi - mini) * (maxj - minj)
    return np.sum(binim) / p

def center_of_gravity (binim):
    m = np.size(binim, 0)
    n = np.size(binim, 1)
    area = np.sum(binim)
    ph = np.sum(binim, 1)
    pv = np.sum(binim, 0)
    X = np.dot(pv, np.array(range(n))) / area
    Y = np.dot(ph, np.array(range(m))) / area
    return X, Y

def centers_of_gravity_slope (binim):
    mini, minj, maxi, maxj = bounding_box(binim)
    im = binim[mini : maxi + 1, minj : maxj + 1]
    n = np.size(im, 1)
    c1 = center_of_gravity(im[:, 0 : int(n / 2)])
    c2 = tuple(map(lambda x, y : x + y, (0, int(n / 2)), center_of_gravity(im[:, int(n / 2) : ])))
    return np.rad2deg(np.arctan2(c2[0] - c1[0], c2[1] - c1[1]))

def edge_points_no (binim):
    res = 0
    m = np.size(binim, 0)
    n = np.size(binim, 1)
    di = [-1, -1, -1, 0, 0, 1, 1, 1]
    dj = [-1, 0, 1, -1, 1, -1, 0, 1]
    for i in range(m):
        for j in range(n):
            count = 0
            for k in range(8):
                ni = i + di[k]
                nj = j + dj[k]
                count += int(ni >= 0 and ni < m and nj >= 0 and nj < n and binim[ni, nj])
            res += int(count == 1)
    return res

def cross_points_no (binim):
    res = 0
    m = np.size(binim, 0)
    n = np.size(binim, 1)
    di = [-1, -1, -1, 0, 0, 1, 1, 1]
    dj = [-1, 0, 1, -1, 1, -1, 0, 1]
    for i in range(m):
        for j in range(n):
            count = 0
            for k in range(8):
                ni = i + di[k]
                nj = j + dj[k]
                count += int(ni >= 0 and ni < m and nj >= 0 and nj < n and binim[ni, nj])
            res += int(count >= 3)
    return res

def extract (binim):
    res = {}
    res["baseline_slant"] = baseline_slant_angle(binim)
    res["aspect_ratio"] = aspect_ratio(binim)
    res["normalized_area"] = normalized_area(binim)
    res["center_x"], res["center_y"] = center_of_gravity(binim)
    res["center_slope"] = centers_of_gravity_slope(binim)
    res["edge_points"] = edge_points_no(binim)
    res["cross_points"] = cross_points_no(binim)
    return res