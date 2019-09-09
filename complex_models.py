import preprocessor as pre
import os
from PIL import Image
import scipy.ndimage.interpolation as ip
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

def augment_image (binim):
    res = []
    for angle in [-15, -10, -5, 0, 5, 10, 15]:
        rotim = ip.rotate(binim, angle, reshape = False)
        res.append(rotim.ravel())
    return res

def fetch_data (path, label, augment):
    X = []
    Y = []
    for image_path in os.listdir(path):
        print(image_path)
        src = int(image_path[9:12])
        wrt = int(image_path[4:7])
        input_path = os.path.join(path, image_path)
        image = Image.open(input_path)
        image = image.resize((50, 50), Image.ANTIALIAS)
        image = np.array(image)
        if augment:
            binims = augment_image(pre.preprocess(image))
            X += binims
            Y += [label for i in range(len(binims))]
        else:
            binim = pre.preprocess(image)
            X.append(binim.ravel())
            Y.append(label)
    
    return X, Y
        
def create_dataset (augment = False):
    X1, Y1 = fetch_data("D:/Diplomski/genuines", 1, augment)
    X2, Y2 = fetch_data("D:/Diplomski/forgeries", 0, augment)
    X = np.array(X1 + X2)
    Y = np.array(Y1 + Y2)
    return X, Y

def train_model (X, Y, state, alpha):
    model = MLPClassifier()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = state)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    y_test = np.array(np.reshape(y_test, len(y_test))).flatten()
    tn, fp, fn, tp = confusion_matrix(y_test, model.predict(x_test), labels = [0, 1]).ravel() / len(y_test)
    print(str(score) + " " + str(tn) + " " + str(fp) + " " + str(fn) + " " + str(tp))
    return model, score, tn, fp, fn, tp
    
X, Y = create_dataset(augment = True)
total_score = 0
total_tn = 0
total_fp = 0
total_fn = 0
total_tp = 0
for i in range(10):
    model, score, tn, fp, fn, tp = train_model(X, Y, i, 0.5)
    total_score += score
    total_tn += tn
    total_fp += fp
    total_fn += fn
    total_tp += tp