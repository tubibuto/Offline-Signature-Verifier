import preprocessor as pre
import feature_extractor as fe
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

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
    for i in range(n):
        x_train, x_test, y_train, y_test = train_test_split(X[i], Y[i], test_size = 0.2, random_state = state)
        model = classifier()
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        print(score)
        models.append(model)
        scores[i] = score
    return models, scores

X, Y = create_dataset()
total_scores = np.zeros(len(X))
for i in range(10):
    models, scores = train_models(X, Y, DecisionTreeClassifier, i)
    total_scores += scores