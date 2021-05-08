import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pydotplus

def main():
    X_train = np.array([[0,0,0,0],[0,0,0,1],[0,1,0,1],[0,1,1,0],[0,0,0,0],[1,0,0,0],[1,0,0,1],[1,1,1,1],[1,0,1,2],[1,0,1,2],[2,0,1,2],[2,0,1,1],[2,1,0,1],[2,1,0,2],[2,0,0,0]])
    y = np.array([0,0,1,1,0,0,0,1,1,1,1,1,1,1,0])

    X_test = np.array([[0,0,1,0],[1,1,0,1],[2,0,1,0]])

    # print(X_train.shape,y.shape,X_test.shape)

    clf = DecisionTreeClassifier()
    clf.fit(X_train,y)
    print(clf.predict(X_test))

    dot_data  = tree.export_graphviz(clf, out_file = None)

    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("D:/workingspace/Github/summary-notebooks-of-postgraduate/统计学习方法/outputs/decision_tree.pdf")

if __name__ == "__main__":
    main()