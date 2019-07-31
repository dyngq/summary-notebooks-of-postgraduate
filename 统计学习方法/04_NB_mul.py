import numpy as np
import matplotlib.pyplot as plt

from concurrent import futures
import time

class Naive_Bayes:
    def __init__(self):
        self.lamb = 0.2
        self.dicts = []
        self.y_dict = []
        self.p_y = []
        self.p_x_y = None
    def fit(self,X_train,y_train):
        X = X_train
        y = y_train
        xy = np.column_stack((X,y))
        
        x_axis = X.shape[1]
        x_types = []

        for i in range(x_axis):
            self.dicts.append(dic(X[:,i]))
            x_types.append(len(self.dicts[i]))

        self.y_dict = dic(y)
        y_types = len(self.y_dict)
        
        for ci in self.y_dict:
            self.p_y.append((len(y[y==ci])+self.lamb) / (len(y)+len(self.y_dict)*self.lamb))

        self.p_x_y = np.zeros(shape=(x_axis,max(x_types),y_types))
        # print(p_x_y)
        for a in range(x_axis):
            for xt,dx in zip(range(x_types[a]),self.dicts[a]):
                # print(xt,dx)
                for yt,dy in zip(range(y_types),self.y_dict):
                    # print(yt,dy)
                    # print(a,xt,yt) # 12组
                    self.p_x_y[a][xt][yt] =   (len(xy[np.multiply(xy[:,-1]==str(dy),xy[:,a]==dx)]) + self.lamb) / (len(y[y==dy])+len(self.dicts[a]) *self.lamb)
                    
                    # print(a+1,dx,dy,len(xy[np.multiply(xy[:,-1]==str(dy),xy[:,a]==dx)]) + self.lamb,(len(y[y==dy])+len(self.dicts[a]) *self.lamb),self.p_x_y[a][xt][yt])
     
        # print(self.p_x_y)

    def predict(self,X_test):
        print('begin')
        y_r_dict = {}
        for di,dd in zip(range(len(self.y_dict)),self.y_dict):
            # print(di,dd)
            y_r_dict[di] = dd

        t = np.ones(shape=(len(self.y_dict)))
        try:
            X_test.shape[1]
        except:
                for yi,ci in zip(range(len(self.y_dict)),self.y_dict):
                    t[yi] = self.p_y[self.y_dict[ci]]

                    for a in range(len(X_test)):
                        t[yi] = t[yi] * self.p_x_y[a][self.dicts[a][X_test[a]]][self.y_dict[ci]]
                        # print(self.p_x_y[a][self.dicts[a][pt[a]]][self.y_dict[ci]])
                        # print(pt,ci,a)

                te = t.tolist()
                print(X_test,te,max(t),y_r_dict[te.index(max(t))] )
        else:
            for pt in X_test:
                for yi,ci in zip(range(len(self.y_dict)),self.y_dict):
                    t[yi] = self.p_y[self.y_dict[ci]]

                    for a in range(X_test.shape[1]):
                        t[yi] = t[yi] * self.p_x_y[a][self.dicts[a][pt[a]]][self.y_dict[ci]]
                        # print(self.p_x_y[a][self.dicts[a][pt[a]]][self.y_dict[ci]])
                        # print(pt,ci,a)

                te = t.tolist()
                print(pt,te,max(t),y_r_dict[te.index(max(t))] )
    
    # 用多进程实现并行，处理多个值的搜索
    def predict_many(self,X_new):
        # 导入多进程
        with futures.ProcessPoolExecutor(max_workers=3) as executor:
            # 建立多进程任务
            tasks=[executor.submit(self.predict,X_new[i]) for i in range(X_new.shape[0])]
            # 驱动多进程运行
            # done_iter=futures.as_completed(tasks)
            # # 提取运行结果
            # res=[future.result() for future in done_iter]
        # return res

def d_one_hot(x,dict):
    for i in x:
        i[1] = dict[i[1]]
    return x

def dic(x):
    num = 0
    dict = {}
    for i in x:
        try:
            dict[i]
        except:
            dict[i] = num
            num = num + 1
        else:
            pass
            
    return dict    


def main():
    X_train = np.array([[1,'S'],[1,'M'],[1,'M'],[1,'S'],[1,'S'],[2,'S'],[2,'M'],[2,'M'],[2,'L'],[2,'L'],[3,'L'],[3,'M'],[3,'M'],[3,'L'],[3,'L']])
    y_train = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])

    clf = Naive_Bayes()
    clf.fit(X_train, y_train)

    # X_test = np.array([2, 'S'])
    X_test = np.array([[2, 'S'],[1, 'M'],[3,'L']])
    # clf.predict(X_test)
    y_predict=clf.predict_many(X_test)
    print(y_predict)
    # print(X_train.shape, y_train.shape)

    # dict = dic(X_train)
    # print(dict)

    # X_train_oh = d_one_hot(X_train,dict)
    # print(X_train_oh)

    


if __name__ == "__main__":
    main()