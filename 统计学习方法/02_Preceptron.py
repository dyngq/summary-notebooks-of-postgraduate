# 已知训练数据集D，其正实例点是 x1=(3,3)T, x2=(4,3)T,负实例点是 x3(1,1)T:
# 用python自编程实现感知机模型，对训练数据集进行分类，并对比分类点选择次序不同对最终结果的影响。

import numpy as np
import matplotlib.pyplot as plt

class Preceptron:
    def  __init__(self):
        self.w = None # 因为还 不知道 X的维度
        self.b = 0
        self.l_rate = 1 # 学习率
    def fit(self, X_train, y_train):
        self.w = np.zeros(X_train.shape[1])
        i = 0
        while i < X_train.shape[0]:
            X = X_train[i]
            y = y_train[i]

            if (y * (np.dot(self.w,X) + self.b) <= 0): # 式子小于0说明是误判点，需要更新w和b
                self.w = self.w + self.l_rate*np.dot(y,X)
                self.b = self.b + self.l_rate*y
                i = 0
            else:
                i = i + 1
            # print(i)
    

def draw(X,w,b):
    #生产分离超平面上的两点
    X_new=np.array([[0], [6]])
    y_predict=-b-(w[0]*X_new)/w[1]

    # print(X_new) # 0 6
    # print(y_predict) # 3 -3
    # 两点确定一条直线，用x=0确定y=3， 用x=6确定y=-3， 所以就是 (0,3) (6,-3) 两点

    #绘制训练数据集的散点图
    plt.plot(X[:2,0],X[:2,1],"g*",label="1")
    plt.plot(X[2:,0], X[2:,1], "rx",label="-1")
    #绘制分离超平面
    plt.plot(X_new,y_predict,"b-")
    #设置两坐标轴起止值
    plt.axis([0,6,0,6])
    #设置坐标轴标签
    plt.xlabel('x1')
    plt.ylabel('x2')
    #显示图例
    plt.legend()
    #显示图像
    plt.show()

def main():
    X_train = np.array([[3,3], [4,3], [1,1]])
    y_train = np.array([1,1,-1])
    # print(X_train.shape[0]) # 3
    # print(X_train.shape[1]) # 2

    clf = Preceptron()
    clf.fit(X_train, y_train)

    print(clf.w)
    print(clf.b)

    draw(X_train, clf.w, clf.b)


if __name__ == "__main__":
    main()