import numpy as np

class Adaboost_d:
    def __init__(self):
        self.w = None
        self.alpha = 0.
        self.clfs = []
    def fit(self,X_train,y_train):
        pass
    def predict(self,X_test):
        pass

from sklearn.ensemble import AdaBoostClassifier 
from sklearn.model_selection import cross_val_score

if __name__ == "__main__":
    X_train = np.array([[0,1,3],[0,3,1],[1,2,2],[1,1,3],[1,2,3],[0,1,2],[1,1,2],[1,1,1],[1,3,1],[0,2,1]])
    y_train = np.array([-1,-1,-1,-1,-1,-1,1,1,-1,-1])
    # print(X_train.shape, y_train.shape)

    clf = Adaboost_d()
    clf.fit(X_train,y_train)
    print(clf.predict(X_train))

    clf2 = AdaBoostClassifier()
    clf2.fit(X_train[:8],y_train[:8])
    print(clf2.predict(X_train[8:]))

    clf3 = AdaBoostClassifier()
    scores = cross_val_score(clf3, X_train, y_train, cv = 5)
    print(scores)