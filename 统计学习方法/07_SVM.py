import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

def draw(X,w,b):
    #生产分离超平面上的两点
    X_new=np.array([[0], [6]])
    y_predict=-b-(w[0]*X_new)/w[1]

    # print(X_new) # 0 6
    # print(y_predict) # 3 -3
    # 两点确定一条直线，用x=0确定y=3， 用x=6确定y=-3， 所以就是 (0,3) (6,-3) 两点

    #绘制训练数据集的散点图
    plt.plot(X[:3,0],X[:3,1],"g*",label="1")
    plt.plot(X[3:,0], X[3:,1], "rx",label="-1")
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

if __name__ == "__main__":
    X_train = np.array([[1,2],[2,3],[3,3],[2,1],[3,2]])
    y_train = np.array([1,1,1,-1,-1])

    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)

    print(clf.coef_)
    print(clf.intercept_)
    draw(X_train,clf.coef_[0],clf.intercept_)


