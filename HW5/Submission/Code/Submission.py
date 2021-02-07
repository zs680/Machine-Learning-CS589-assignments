from run_me import original_image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import copy
import sys



def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data


List=[]
for i in range(100):
    image=load_image('../../Data/Faces/face_{}.png'.format(i))
    image=image.reshape(2500)
    List.append(image)

X=np.array(List)

from numpy import linalg as la
B=np.matmul(np.transpose(X),X)
w, v = la.eigh(B)

W=np.array(w)
V=np.array(v)

from numpy import newaxis
#W=W[:,newaxis]


#covariance matrix of the given data is with mean 0

E=np.vstack([W,V])

E_n=E[:,E[0,:].argsort()]
List=[3, 5, 10, 30, 50, 100, 150, 300]
j=0

"""for row in X:
    for i in List:
        E1=E_n[1:2501,2499-i+1:2500]
        Y=np.transpose(E1)
        Sum=np.zeros(Y[0,:].shape)
        for row_1 in Y:
            y=np.dot(row,row_1)
            Sum=np.add(Sum,row_1*y)
        Sum=Sum.reshape((50,50))
        Sum=Sum.astype(np.uint8)
        image = Image.fromarray(Sum)
        image.save('../../Submission/Figures/face{facename} dim{list}.png'.format(facename =j, list=i))
        #image.show()
    row=row.reshape((50,50))
    row=row.astype(np.uint8)
    image1=Image.fromarray(row)
    image1.save('../../Submission/Figures/face {}.png'.format(j))
    j+=1"""

image=load_image( '../../Data/shopping-street.jpg')
Z = image.reshape((-1,3))
#print(vector.shape)

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

lst = [2, 5, 10, 25, 50, 100, 200]
distortions = []
for i in lst:
    kmeans = KMeans(n_clusters=i, random_state=0).fit(Z)
    Clusters=np.asarray(kmeans.cluster_centers_,dtype= np.uint8)
    Labels=np.asarray(kmeans.labels_,dtype=np.uint8)
    Labels=Labels.reshape(image.shape[0],image.shape[1])
    array = np.zeros([image.shape[0], image.shape[1], 3], dtype=np.uint8)
    #for j in range(i):
        #array[Labels==j]=Clusters[j]
    #new_image=Image.fromarray(array)
    #new_image.save("../../Submission/shopping-street{}.png".format(i))
    distortions.append(sum(np.min(cdist(Z, Clusters, 'euclidean'), axis=1)) / Z.shape[0])

print(distortions)


# Plot the elbow
plt.plot(lst, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
