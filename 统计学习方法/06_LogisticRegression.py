import numpy as np
from sklearn.linear_model import LogisticRegression as LR


def sigmoid(z):
    sigmoid = 1.0 / (1.0 + np.exp(-z))
    return sigmoid

class LogisticRegression:
    def __init__(self,lr=0.1,epoch=3333):
        self.w = None
        self.b = 0
        self.lr = lr
        self.epoch = epoch
    def fit(self,X_train,y_train):
        self.w = np.zeros(X_train.shape[1])
        self.w=np.array([[0]*X_train.shape[1]],dtype=np.float)
        # print(self.w)
        self.b = 0
        for i in range(self.epoch):
            # print(X_train.T*(y_train-sigmoid(np.dot(self.w,X_train.T)+self.b)))
            # print(np.dot(X_train,self.w.T)+self.b)
            t = X_train*(y_train.T-sigmoid(np.dot(X_train,self.w.T)+self.b))
            # print(X_train,self.w.T)
            # t = t[0]
            # print(t)
            # print(np.sum(t,axis=0))
            # print()
            a = t.sum(axis=0)
            # print(a)
            # print(a.shape())
            # print(self.w.shape)
            self.w = self.w + (self.lr*a)
            self.b = self.b + self.lr*np.sum(y_train.T-sigmoid(np.dot(X_train,self.w.T)+self.b),axis=0)
            # print(self.b)

            # print(a)
    def predict(self,X_test):
        if(sigmoid(np.dot(X_test,self.w.T))>=0.5):
            return 1
        else:
            return 0

if __name__ == "__main__":
    X_train = np.array([[3,3,3],[4,3,2],[2,1,2],[1,1,1],[-1,0,1],[2,-2,1]])
    y_train = np.array([[1,1,1,0,0,0]])

    clf = LogisticRegression()
    clf.fit(X_train,y_train)

    # clf2 = LR()
    # clf2.fit(X_train,y_train)

    X_test = np.array([1,2,-2])

    print(clf.predict(X_test))
    # print(clf2.predict(X_test))

    a = np.array([[1,1,1],[2,3,4]])
    print(np.sum(a,axis=0))