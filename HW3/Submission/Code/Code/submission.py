from run_me import train_s_x, train_s_y, test_s_x, test_s_y
import numpy as np
import operator as op
from functools import reduce
from run_me import compute_MSE
#python 3.8
#from math import comb
#comb(10,3)

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

def Phi_1(x,p):
    List=[]
    for j in range(p+1):
        List.append((ncr(p,j)**.5)*x**j)
    return np.array(List)
delta=.5
def Phi_2(X,p):
    List=[]
    for x in X:
        list=[]
        for j in range(p+1):
            if j==0:
                list.append(1)
            else:
                list.append(np.sin(j*delta*x))
                list.append(np.cos(j*delta*x))
        List.append(list)
    return np.transpose(np.array(List))

def basis_expansion(Phi,train_x,p):
    basis_expansion=Phi(train_x,p)
    return np.transpose(basis_expansion)


#print(basis_expansion(Phi_1,train_s_x,3).shape)
#print(basis_expansion(Phi_2,train_s_x,3).shape)

#print(basis_expansion(Phi_2,train_s_x,3))
from sklearn.linear_model import Ridge
def BERR_pred(train_x,train_y,test_x,p,Phi):
    clf = Ridge(alpha=.1)
    clf.fit(basis_expansion(Phi,train_x,p), train_y)
    prediction=clf.predict(basis_expansion(Phi,test_x,p))
    return prediction

#print(BERR_pred(train_s_x,train_s_y,test_s_x,3,Phi_2))



def kernel_matrix_1(x_1,x_2,p):
    return (1+np.dot(x_1,x_2))**p

def kernel_matrix_2(x_1,x_2,p):
    sum=0
    for j in range(p+1):
        if j==0:
            sum+=1
        else:
            sum+=np.cos(j*delta*x_1)*np.cos(j*delta*x_2)+np.sin(j*delta*x_1)*np.sin(j*delta*x_2)
    return sum



def ker(X,p,kernel_matrix):
    N=X.shape[0]
    K=np.zeros((N,N))
    for i in range(0,N):
        for j in range(0,N):
            K[i][j]=kernel_matrix(X[i],X[j],p)
    return K

def K(train_x,X,p,kernel_matrix):
    N=train_x.shape[0]
    K=np.zeros(N)
    for i in range(0,N):
        K[i]=kernel_matrix(train_x[i],X,p)
    return K







#Kernel Ridge Regression


from numpy.linalg import inv
def KRRS(train_x,train_y,X,p,kernel_matrix):
    N=train_x.shape[0]
    Id=np.identity(N)
    alpha=np.dot(inv(ker(train_x,p,kernel_matrix)+.1*Id),train_y)
    y_hat=np.dot(alpha,K(train_x,X,p,kernel_matrix))
    return y_hat

def KRRS_pred(train_x,train_y,test_x,p,kernel_matrix):
    prediction=np.zeros(test_s_x.shape[0])
    for i in range(test_s_x.shape[0]):
        prediction[i]=KRRS(train_x,train_y,test_x[i],p,kernel_matrix)
    return prediction

#print(KRRS_pred(train_s_x,train_s_y,test_s_x,2,kernel_matrix_1))
#print(KRRS_pred(train_s_x,train_s_y,test_s_x,3,kernel_matrix_2))
A_1={1,2,4,6}
A_2={3,5,10}


train_x=train_s_x
train_y=train_s_y
test_x=test_s_x
test_y=test_s_y
"""for p in A_1:
    print("MSE_BERR_Phi_1 {}".format(p),compute_MSE(BERR_pred(train_x,train_y,test_x,p,Phi_1),test_s_y))

for p in A_1 :
    print("MSE_KRRS_kernel_matrix_1 {}".format(p), compute_MSE(KRRS_pred(train_x,train_y,test_x,p,kernel_matrix_1),test_s_y))

for p in A_2:
    print("MSE_BERR_Phi_2 {}".format(p),compute_MSE(BERR_pred(train_x,train_y,test_x,p,Phi_2),test_s_y))

for p in A_2 :
    print("MSE_KRRS_kernel_matrix_2 {}".format(p), compute_MSE(KRRS_pred(train_x,train_y,test_x,p,kernel_matrix_2),test_s_y))
"""

"""import matplotlib.pyplot as plt
x=test_s_x
y=test_s_y
prediction=BERR_pred(train_s_x,train_s_y,x,10,Phi_2)
plt.scatter(x,y)
plt.scatter(x,prediction)
#plt.show()

import matplotlib.pyplot as plt
x=test_s_x
y=test_s_y
prediction=KRRS_pred(train_s_x,train_s_y,x,10,kernel_matrix_2)
plt.scatter(x,y)
plt.scatter(x,prediction)
plt.show()"""

from sklearn.kernel_ridge import KernelRidge
from run_me import train_h_x, train_h_y, test_h_x
import numpy as np
X=train_h_x
print(X.shape)
y=train_h_y
print(y.shape)
clf = KernelRidge(alpha=1.0,kernel="rbf",gamma=1 )
#KernelRidge(alpha=1.0,kernel="polynomial",degree=3)
#KernelRidge(alpha=1.0)#linear
clf.fit(X, y)
predict=clf.predict(test_h_x)

print(predict)

from run_me import train_c_x, train_c_y, test_c_x

from sklearn import svm


gamma=1
models = [svm.SVC(kernel='linear', C=.5),
          svm.SVC(kernel='rbf', gamma=gamma, C=.5),
          svm.SVC(kernel='poly', degree=3, gamma=gamma, C=.5),
          svm.SVC(kernel='poly', degree=5, gamma=gamma, C=.5),
          svm.SVC(kernel='linear', C=.05),
          svm.SVC(kernel='rbf', gamma=gamma, C=.05),
          svm.SVC(kernel='poly', degree=3, gamma=gamma, C=.05),
          svm.SVC(kernel='poly', degree=5, gamma=gamma, C=.05),
          svm.SVC(kernel='linear', C=.0005),
          svm.SVC(kernel='rbf', gamma=gamma, C=.0005),
          svm.SVC(kernel='poly', degree=3, gamma=gamma, C=.0005),
          svm.SVC(kernel='poly', degree=5, gamma=gamma, C=.0005),
          svm.SVC(kernel='rbf', C = 0.5, gamma = 0.01),
          svm.SVC(kernel='poly', degree=3,C = 0.5, gamma = 0.01),
          svm.SVC(kernel='poly', degree=5,C = 0.5, gamma = 0.01),
          svm.SVC(kernel='rbf', C = 0.5, gamma = 0.001),
          svm.SVC(kernel='poly', degree=3,C = 0.5, gamma = 0.001),
          svm.SVC(kernel='poly', degree=5,C = 0.5, gamma = 0.001),
          svm.SVC(kernel='rbf', C = 0.05, gamma = 0.01),
          svm.SVC(kernel='poly', degree=3,C = 0.05, gamma = 0.01),
          svm.SVC(kernel='poly', degree=5,C = 0.05, gamma = 0.01),
          svm.SVC(kernel='rbf', C = 0.05, gamma = 0.001),
          svm.SVC(kernel='poly', degree=3,C = 0.05, gamma = 0.001),
          svm.SVC(kernel='poly', degree=5,C = 0.05, gamma = 0.001),
          svm.SVC(kernel='rbf', C = 0.0005, gamma = 0.001),
          svm.SVC(kernel='poly', degree=3,C = 0.0005, gamma = 0.001),
          svm.SVC(kernel='poly', degree=5,C = 0.0005, gamma = 0.001)
          ]

#models = (clf.fit(X, y) for clf in models)
from CV import find_best
from run_me import train_c_x, train_c_y
find_best(models, train_c_x, train_c_y,5)

#models = (clf.fit(X, y) for clf in models)
