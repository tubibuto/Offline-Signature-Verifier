import preprocessor as pre
import feature_extractor as fe
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix

def fetch_data (path):
    n = 0
    for image_path in os.listdir(path):
        n = np.max([n, int(image_path[0:3])])
        
    X = [[] for i in range(n)]
    Y = [[] for i in range(n)]
    for image_path in os.listdir(path):
        print(image_path)
        src = int(image_path[5:8]) - 1
        wrt = int(image_path[0:3]) - 1
        input_path = os.path.join(path, image_path)
        image = mpimg.imread(input_path)
        binim = pre.preprocess(image)
        feats = list(fe.extract(binim).values())
        X[src].append(feats)
        Y[src].append(int(src == wrt))
    
    return X, Y

def create_dataset ():
    X1, Y1 = fetch_data("D:/Diplomski/Dataset/dataset1/real")
    X2, Y2 = fetch_data("D:/Diplomski/Dataset/dataset1/forge")
    
    n = len(X1)
    X = [[] for i in range(n)]
    Y = [[] for i in range(n)]
    for i in range(n):
        X[i] = X1[i] + X2[i]
        Y[i] = Y1[i] + Y2[i]
        
    for i in range(n):
        X[i] = np.matrix(X[i])
        Y[i] = np.transpose(np.matrix(Y[i]))
        
    return X, Y
    
def train_models (X, Y, classifier, state):
    n = len(X)
    models = []
    scores = np.zeros(n)
    tns = np.zeros(n)
    fps = np.zeros(n)
    fns = np.zeros(n)
    tps = np.zeros(n)
    for i in range(n):
        x_train, x_test, y_train, y_test = train_test_split(X[i], Y[i], test_size = 0.2, random_state = state)
        model = classifier()
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        y_test = np.array(np.reshape(y_test, len(y_test))).flatten()
        tn, fp, fn, tp = confusion_matrix(y_test, model.predict(x_test), labels = [0, 1]).ravel() / len(y_test)
        print(str(score) + " " + str(tn) + " " + str(fp) + " " + str(fn) + " " + str(tp))
        models.append(model)
        scores[i] = score
        tns[i] = tn
        fps[i] = fp
        fns[i] = fn
        tps[i] = tp
        
    return models, scores, tns, fps, fns, tps

X, Y = create_dataset()
total_scores = np.zeros(len(X))
total_tns = np.zeros(len(X))
total_fps = np.zeros(len(X))
total_fns = np.zeros(len(X))
total_tps = np.zeros(len(X))
for i in range(10):
    models, scores, tns, fps, fns, tps = train_models(X, Y, LogisticRegression, i)
    total_scores += scores
    total_tns += tns
    total_fps += fps
    total_fns += fns
    total_tps += tps