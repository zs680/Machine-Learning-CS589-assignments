from PIL import Image
# Open the image form working directory
"""image = Image.open('../../Data/shopping-street.jpg')
# summarize some details about the image
print(image)
print(image.format)
print(image.size)
print(image.mode)
# show the image
image.show()"""
# load and display an image with Matplotlib
"""from matplotlib import image
from matplotlib import pyplot
# load image as pixel array
image = image.imread('../../Data/shopping-street.jpg')
# summarize shape of the pixel array
print(image.dtype)
print(image.shape)
print(image)
# display the array of pixels as an image
pyplot.imshow(image)
pyplot.show()"""
"""from PIL import Image
from numpy import asarray
# load the image
image = Image.open('../../Data/shopping-street.jpg')
# convert image to numpy array
data = asarray(image)
print(data)
print(image)
print(type(data))
# summarize shape
print(data.shape)

# create Pillow image
image2 = Image.fromarray(data)
print(image2)
image.show()
print(type(image2))
# summarize image details
print(image2.mode)
print(image2.size)"""

"""import numpy as np
from PIL import Image

im = np.array(Image.open('../../Data/shopping-street.jpg').convert('L')) #you can pass multiple arguments in single line
print(type(im))

gr_im= Image.fromarray(im).save('gr_kolala.png')"""

#playing with numpy array
"""import numpy as np
A=np.arange(6).reshape((3, 2))
print(A)

from numpy import linalg as LA
B=np.matmul(np.transpose(A),A)
print(B)
C=np.matmul(A,np.transpose(A))
print(C)
D=np.array([[ 4, 3,  5,1],
 [ 3, 13, 23,7],
 [ 1, 23, 41,8]])
print(D)
#H = D[D[0,:].argsort()]
#print(H)
H=D[:,D[0,:].argsort()]
print(H)
print(H[:,1:4])

#find the index of rows in numpy array
print(np.where((D==(4, 3,  5,1)).all(axis=1))[0])
j=0
for row in D:
  print(j)
  j+=1"""

#k mean elbow
"""from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])

plt.plot()
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('Dataset')
plt.scatter(x1, x2)
plt.show()

# create new plot and data
plt.plot()
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
print(X)
colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']

# k means determine k
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    #print(kmeanModel.cluster_centers_)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()"""


#creating RGB image
import numpy as np
from PIL import Image

array = np.zeros([100, 200, 3], dtype=np.uint8)
array[:,:100] = [255, 128, 0] #Orange left side
array[:,100:] = [0, 0, 255]   #Blue right side

img = Image.fromarray(array)
img.save('testrgb.png')

"""import numpy as np
from PIL import Image

array = np.zeros([100, 200, 4], dtype=np.uint8)
array[:,:100] = [255, 128, 0, 255] #Orange left side
array[:,100:] = [0, 0, 255, 255]   #Blue right side

# Set transparency depending on x position
for x in range(200):
    for y in range(100):
        array[y, x, 3] = x

img = Image.fromarray(array)
img.save('testrgba.png')"""
