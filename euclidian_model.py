import preprocessor as pre
import feature_extractor as fe
import os

class euclid_model:
    # takes dataset path and calculates all model's stats based on images from it
    def __init__ (self, path):
        # determine number of distinct signatures
        self.n = 0
        for image_path in os.listdir(path):
            self.n = np.max([self.n, int(image_path[0:3])])
        # init image feature list
        image_feats = \
            [{"baseline_slant" : np.array([]), \
              "aspect_ratio" : np.array([]), \
              "normalized_area" : np.array([]), \
              "center_x" : np.array([]), \
              "center_y" : np.array([]), \
              "center_slope" : np.array([]), \
              "edge_points" : np.array([]), \
              "cross_points" : np.array([]), } for i in range(self.n)]
        # populate image feature list
        for image_path in os.listdir(path):
            print(image_path)
            i = int(image_path[0:3]) - 1
            input_path = os.path.join(path, image_path)
            image = mpimg.imread(input_path)
            binim = pre.preprocess(image)
            feats = fe.extract(binim)
            for feat_key in feats:
                image_feats[i][feat_key] = np.append(image_feats[i][feat_key], feats[feat_key])
        # init signature stats list
        self.signature_stats = \
            [{"baseline_slant" : {"mean" : 0, "std" : 0}, \
              "aspect_ratio" : {"mean" : 0, "std" : 0}, \
              "normalized_area" : {"mean" : 0, "std" : 0}, \
              "center_x" : {"mean" : 0, "std" : 0}, \
              "center_y" : {"mean" : 0, "std" : 0}, \
              "center_slope" : {"mean" : 0, "std" : 0}, \
              "edge_points" : {"mean" : 0, "std" : 0}, \
              "cross_points" : {"mean" : 0, "std" : 0}, \
              "distance" : {"mean" : 0, "std" : 0, "max" : 0}} for i in range(self.n)]
        # populate signature stats list
        for i in range(self.n):
            feats = image_feats[i]
            for feat_key in feats:
                self.signature_stats[i][feat_key]["mean"] = np.mean(feats[feat_key])
                self.signature_stats[i][feat_key]["std"] = np.std(feats[feat_key])
        # populate average distance stat
        for i in range(self.n):
            feats = image_feats[i]
            m = np.size(feats["baseline_slant"])
            dists = np.zeros(m)
            for j in range(m):
                temp_feats = {}
                for feat_key in feats:
                    temp_feats[feat_key] = feats[feat_key][j]
                dists[j] = self.distance(i, temp_feats)
            print(dists)
            self.signature_stats[i]["distance"]["mean"] = np.mean(dists)
            self.signature_stats[i]["distance"]["std"] = np.std(dists)
            self.signature_stats[i]["distance"]["max"] = np.max(dists)
        # set distance thresholds
        self.thresholds = [{"std1" : 0, "std2" : 0, "std3" : 0, "max" : 0} for i in range(self.n)]
        for i in range(self.n):
            self.thresholds[i]["std1"] = \
                self.signature_stats[i]["distance"]["mean"] + self.signature_stats[i]["distance"]["std"]
            self.thresholds[i]["std2"] = \
                self.signature_stats[i]["distance"]["mean"] + 2 * self.signature_stats[i]["distance"]["std"]
            self.thresholds[i]["std3"] = \
                self.signature_stats[i]["distance"]["mean"] + 3 * self.signature_stats[i]["distance"]["std"]
            self.thresholds[i]["max"] = self.signature_stats[i]["distance"]["max"]

    def distance (self, i, feats):
        res = 0
        for feat_key in feats:
            res += \
                ((feats[feat_key] - self.signature_stats[i][feat_key]["mean"]) / self.signature_stats[i][feat_key]["std"]) ** 2
        return res / self.n
    
    def is_valid (self, i, feats, threshold_type = "std2"):
        dist = self.distance(i, feats)
        return dist <= self.thresholds[i][threshold_type]

# training
model = euclid_model("D:/Diplomski/Dataset/dataset1/real")
        
# test
def test (model):
    path = "D:/Diplomski/Dataset/dataset1/real"
    res = np.array([])
    for image_path in os.listdir(path):
        print(image_path)
        i = int(image_path[5:8]) - 1
        input_path = os.path.join(path, image_path)
        image = mpimg.imread(input_path)
        binim = pre.preprocess(image)
        feats = fe.extract(binim)
        res = np.append(res, model.is_valid(i, feats, "max"))
    return res