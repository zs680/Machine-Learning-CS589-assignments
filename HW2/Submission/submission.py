import pandas as pd
import run_me
from run_me import split_train_test
from run_me import train_x,train_y,test_x
from run_me import train_knn_x,train_knn_y,test_knn_x
from run_me import compute_error
from run_me import Average
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier



A = [3, 6, 9, 12, 14]
lst=[]
for i in A:
    List=[]
    for train_index, test_index in split_train_test(len(train_x),5):
            train_x_CV, train_y_CV,test_x_CV,test_y_CV = train_x[train_index], train_y[train_index],train_x[test_index],train_y[test_index]
            #print(train_x_CV, train_y_CV,test_x_CV,test_y_CV)
            Model=DecisionTreeClassifier(max_depth=i)
            Model.fit(train_x_CV, train_y_CV)
            test_y_hat=Model.predict(test_x_CV)
            error=compute_error(test_y_hat,test_y_CV)
            List.append(error)
    average_error=Average(List)
    lst.append(average_error)

print(lst)

def find_error(Model,train_x_CV,train_y_CV,test_x_CV,test_y_CV):
    model=Model
    model.fit(train_x_CV, train_y_CV)
    test_y_hat=model.predict(test_x_CV)
    return compute_error(test_y_hat,test_y_CV)


def find_best(A,train_x, train_y,k):
    lst=[]
    for i in A:
        List=[]
        for train_index, test_index in split_train_test(len(train_x),k):
            train_x_CV, train_y_CV,test_x_CV,test_y_CV = train_x[train_index], train_y[train_index],train_x[test_index],train_y[test_index]
            Model=DecisionTreeClassifier(max_depth=i)
            error=find_error(Model, train_x_CV, train_y_CV, test_x_CV, test_y_CV)
            List.append(error)
        average_error=Average(List)
        lst.append(average_error)

    for j in range(0,len(A)):
        if lst[j]==min(lst):
            print(A[j])
            a=A[j]
    model=DecisionTreeClassifier(max_depth=a)
    model.fit(train_x,train_y)
    #prediction=predict(test_x)
    #return prediction


A = [3, 6, 9, 12, 14]
"""Model=[]
for i in A:
    Model.append(DecisionTreeClassifier(max_depth=i))"""

print(find_best(A,train_x,train_y,5))
