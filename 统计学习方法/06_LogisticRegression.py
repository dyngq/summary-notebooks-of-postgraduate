import numpy as np
from sklearn.linear_model import LogisticRegression as LR


def sigmoid(z):
    sigmoid = 1.0 / (1.0 + np.exp(-z))
    return sigmoid

class LogisticRegression:
    def __init__(self,lr=0.1,epoch=100):
        self.w = None
        self.b = 0
        self.lr = lr
        self.epoch = epoch
    def fit(self,X_train,y_train):
        self.w = np.zeros(X_train.shape[1])
        # print(self.w.shape)
        self.b = 0
        for i in range(self.epoch):
            self.w = self.w + (self.lr*np.dot(X_train.T,(y_train-sigmoid(np.dot(self.w,X_train.T)+self.b))))/X_train.shape[0]
            self.b = self.b + (y_train-sigmoid(np.dot(self.w,X_train.T)+self.b))/X_train.shape[0]
    def predict(self,X_test):
        if(sigmoid(np.dot(self.w,X_test[0]))>=0.5):
            return 1
        else:
            return 0

if __name__ == "__main__":
    X_train = np.array([[3,3,3],[4,3,2],[2,1,2],[1,1,1],[-1,0,1],[2,-2,1]])
    y_train = np.array([1,1,1,0,0,0])

    clf = LogisticRegression()
    clf.fit(X_train,y_train)

    clf2 = LR()
    clf2.fit(X_train,y_train)

    X_test = np.array([[1,2,-2]])

    print(clf.predict(X_test))
    print(clf2.predict(X_test))
