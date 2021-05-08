from numpy import *


def loadDataSet():
    dataSet=[[0,1,3],[0,3,1],[1,2,2],[1,1,3],[1,2,3],[0,1,2],[1,1,2],[1,1,1],[1,3,1],[0,2,1]]
    label=[1,1,1,-1,-1,-1,1,1,1,-1]
    return mat(dataSet).T,mat(label)


def adaBoostTrain(dataSet,label,numIt=10):
    classifDict=[]
    m=shape(dataSet)[0]
    totalRetResult=mat(zeros((m,1)))
    weight=mat(ones((m,1))/m)
    for i in range(numIt):
        bestFeat,error,EstClass=decisionTree(dataSet,label,weight)
        alpha=float(0.5*log((1-error)/error))
        bestFeat['alpha']=alpha
        classifDict.append(bestFeat)
        wtx=multiply(-1*alpha*mat(label).T,EstClass)
        weight=multiply(weight,exp(wtx))
        weight=weight/sum(weight)
        totalRetResult += alpha*EstClass
        totalError = (sum(label.T != sign(totalRetResult))) / float(m)
        if totalError==0:break
    return classifDict,totalRetResult

def splitDataSet(dataMat,feat,value,comp,m):
    retArray=ones((m,1))
    if comp=='LT':
        retArray[dataMat[:,feat] <value] = -1.0
    else:
        retArray[dataMat[:,feat] >value] = -1.0
    return  retArray

def decisionTree(dataSet,labelList,weight):
    dataMat=mat(dataSet);labelMat=mat(labelList).T
    bestFeat={}
    minError=inf
    m,n=shape(dataMat)
    bestClass=mat(zeros((m,1)))
    for i in range(n):
        sortedIndex=argsort(dataMat,axis=i)
        for j in range(m-1):
            value=(dataMat[sortedIndex[j],i]+dataMat[sortedIndex[j+1],i])/2.0
            for comp in ['LT', 'ST']:  # 符号可以是大于或者小于 LT:larger than    ST:small than
                retArray=splitDataSet(dataMat,i,value,comp,m)
                errSet=mat(ones((m,1)))
                errSet[retArray == labelMat] =0
                #print D,errSet
                weightError=weight.T*errSet
                #print weightError
                if weightError<minError:
                    minError=weightError
                    bestFeat['feat']=i
                    bestFeat['value']=value
                    bestFeat['comp']=comp
                    bestClass=retArray.copy()
    return bestFeat,minError,bestClass

dataSet,label=loadDataSet()
classifDict,totalRetResult=adaBoostTrain(dataSet,label)
print ("classifDict",classifDict)
print (sign(totalRetResult))