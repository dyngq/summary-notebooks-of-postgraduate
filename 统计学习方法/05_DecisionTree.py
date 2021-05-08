import numpy as np

class BaseDT:
    def  __init__(self):
        pass
    def fit(self,X_train,y_train,w):
        pass


if __name__ == "__main__":
    X_train = np.array([[0,1,3],[0,3,1],[1,2,2],[1,1,3],[1,2,3],[0,1,2],[1,1,2],[1,1,1],[1,3,1],[0,2,1]])
    y_train = np.array([-1,-1,-1,-1,-1,-1,1,1,-1,-1])
    # print(X_train.shape, y_train.shape)

    clf = BaseDT()
    w = np.array([1.0/X_train.shape[0]]*X_train.shape[0],dtype=np.float)
    # print(w)
    clf.fit(X_train,y_train,w)
    # print(clf.predict(X_train))