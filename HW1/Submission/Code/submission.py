import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from run_me import train_x,train_y,test_x
from run_me import compute_error
from run_me import split_train_test
from sklearn.model_selection import cross_val_score


from sklearn import neighbors

import time
import math
import os
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt





Time=[]
List=[]
A=[3,6,9,12,15]
for i in A:
    start = time.time()
    regressor = DecisionTreeRegressor(random_state=0,max_depth=i)
    scores = cross_val_score(regressor,train_x, train_y,scoring="neg_mean_absolute_error", cv=5)
    rmse_scores = np.mean(-scores)
    end = time.time()
    Time.append(end-start)
    List.append(rmse_scores)

#configuration/model with the lowest out of sample error
# #predict the target values for the provided test set
for k in range(0,len(A)):
    if List[k]==min(List):
        print(A[k])
        regressor = DecisionTreeRegressor(random_state=0, max_depth=A[k])
        regressor.fit(train_x,train_y)
        #print(regressor.predict(test_x))



"""
plt.plot(A,Time)
plt.ylabel('tree depth')
plt.ylabel('time')
plt.show()
plt.savefig('Time.png')
"""



#python code for k-fold cross validation error



def Average(lst):
    return sum(lst) / len(lst)



#td = np.load('../../Data/dataA/train.npy')

n_neighbors=[3, 5, 10, 20, 25]
List2=[]
List3=[]
for i in n_neighbors:
    for train_index, test_index in split_train_test(len(train_x),5):
    #print("TRAIN:", train_index, "TEST:", test_index)
        train_x_kf, train_y_kf,test_x_kf,test_y_kf = train_x[train_index], train_y[train_index],train_x[test_index],train_y[test_index]
        knn=neighbors. KNeighborsRegressor(i,weights="uniform")
        knn.fit(train_x_kf, train_y_kf)
        test_y_hat=knn.predict(test_x_kf)
        List2.append(compute_error(test_y_hat,test_y_kf))
    List3.append(Average(List2))
#print((min(List3)))
#computation of the test error
for k in range(0,len(n_neighbors)):
    if List3[k]==min(List3):
        print(n_neighbors[k])
        knn = neighbors.KNeighborsRegressor(n_neighbors[k],weights="uniform")
        knn.fit(train_x,train_y)
        #knn.predict(test_x))
List5=[]
List4=[]
from sklearn.linear_model import Ridge
Alpha=[10**-6,10**-4,10**-2,1,10**2]
for alpha in A:

    for train_index, test_index in split_train_test(len(train_x), 5):
        # print("TRAIN:", train_index, "TEST:", test_index)
        train_x_kf, train_y_kf, test_x_kf, test_y_kf = train_x[train_index], train_y[train_index], train_x[test_index], \
                                                       train_y[test_index]
        knn = Ridge(alpha)
        knn.fit(train_x_kf, train_y_kf)
        test_y_hat = knn.predict(test_x_kf)
        List5.append(compute_error(test_y_hat, test_y_kf))
    List4.append(Average(List5))
    # print((min(List3)))
    # computation of the test error
for k in range(0, len(Alpha)):
    if List4[k] == min(List4):
        print(Alpha[k])
        knn = Ridge(Alpha[k])
        knn.fit(train_x, train_y)