import time
import numpy as np


#.splite()
"""text = 'geeks for geeks'

# Splits at space
print(text.split())

word = 'geeks, for, geeks'

# Splits at ','
print(word.split(', '))

word = 'geeks:for:geeks'

# Splitting at ':'
print(word.split(':'))

word = 'CatBatSatFatOr'

# Splitting at 3
print([word[i:i+3] for i in range(0, len(word), 3)])"""

#writing and reading a text file
"""N = 100  # 10.000.000 datapoints
with open('data.txt', 'w') as data:
    for _ in range(N):
        data.write(str(10*np.random.random())+',')

start = time.time()

with open('data.txt', 'r') as data:
    string_data = data.read()

list_data = string_data.split(',')
print("List_data",list_data)
list_data.pop()
data_array = np.array(list_data, dtype=float).reshape(10, 10)

end = time.time()

print("### 10 million points of data ###")
print("\nData summary:\n", data_array)
#print("\nData shape:\n", data_array.shape)
#print(f"\nTime to read: {round(end-start,5)} seconds.")
np.save('data.npy', data_array)"""



import functools
import operator
List=[i for i in range(1,21)]
print(List)

lis = [ 1 , 3, 5, 6, 2, ]
print ("The sum of the list elements is : ",end="")
print (functools.reduce(operator.add,lis))
print ("The product of list elements is : ",end="")
print (functools.reduce(operator.mul,List))
print ("The concatenated product is : ",end="")
print (functools.reduce(operator.add,["geeks","for","geeks"]))


import itertools

# importing functools for reduce()
import functools

# initializing list
lis = [ 1, 3, 4, 10, 4 ]

# priting summation using accumulate()
print ("The summation of list using accumulate is :",end="")
print (list(itertools.accumulate(lis,lambda x,y : x+y)))

# priting summation using reduce()
print ("The summation of list using reduce is :",end="")
print (functools.reduce(lambda x,y:x+y,lis))

print(functools.reduce(operator.mul, range(5, 3, -1)))


def ncr(n, r):
    r = min(r, n-r)
    numer = functools.reduce(operator.mul, range(n, n-r, -1), 1)
    denom = functools.reduce(operator.mul, range(1, r+1), 1)
    return numer / denom

def Phi_notational(i):
    string=""
    for j in range(i+1):
        string=string+"{}.x^{}+".format(ncr(i,j)**.5,j)
    return string


print(Phi_notational(5))

import numpy
l=numpy.array([1,2,0,3,4])
print(l.shape[0])
print(l[0])
print(l.shape)
delta=.5
def kernel_matrix(x_1,x_2,p):
    sum=0
    for j in range(p+1):
        if j==0:
            sum+=1
        else:
            sum+=np.cos(j*delta*x_1)*np.cos(j*delta*x_2)+np.sin(j*delta*x_1)*np.sin(j*delta*x_2)
    return sum

print(kernel_matrix(np.pi,np.pi/3,1))

sum=0
for i in range(4):
    sum+=i
    print(sum)
print(sum)
delta=.5
def Phi(x,p):
    List=[]
    for j in range(p+1):
        if j==0:
            List.append(1)
        else:
            List.append(np.sin(j*delta*x))
            List.append(np.cos(j*delta*x))

    return np.array(List)
print("HahA",Phi(np.pi,2))


from run_me import train_c_x, train_c_y, test_c_x
from sklearn import svm
import pandas as pd
model=svm.SVC()
model.fit(train_c_x,train_c_y)
y_hat=model.predict(test_c_x)
print(y_hat)
predicted_h_y = np.random.randint(-1, 1, test_c_x.shape[0])

print(predicted_h_y)
import pandas as pd
df=pd.DataFrame()
df["y"]=predicted_h_y
df["y_hat"]=y_hat
print(df.query('y!=y_hat').shape[0])

    #return df.query('y!=y_hat')
"""def compute_error():
    from sklearn import svm
    import pandas as pd
    model=svm.SVC()
    model.fit(train_s_x,train_s_y)
    y_hat=model.predict(test_s_x)
    #print(y_hat)
    df=pd.dataframe()
    df["y"]=test_s_y
    df["y_hat"]=y_hat
    return df.query('y!=y_hat')
    #return df.query('y!=y_hat')"""


