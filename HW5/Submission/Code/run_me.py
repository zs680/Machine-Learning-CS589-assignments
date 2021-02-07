import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import copy
import sys


def visualize(im1, im2, k):
    # displays two images
    im1 = im1.astype('uint8')
    im2 = im2.astype('uint8')
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(im1)
    plt.axis('off')
    plt.title('Original')
    f.add_subplot(1, 2, 2)
    plt.imshow(im2)
    plt.axis('off')
    plt.title('Cluster: ' + str(k))
    plt.savefig('k_means_' + str(k) + '.jpg')
    plt.show()
    return None


def MSE(Im1, Im2):
    # computes error
    Diff_Im = Im2 - Im1
    Diff_Im = np.power(Diff_Im, 2)
    Diff_Im = np.sum(Diff_Im, axis=2)
    Diff_Im = np.sqrt(Diff_Im)
    sum_diff = np.sum(np.sum(Diff_Im))
    avg_error = sum_diff / float(Im1.shape[0] * Im2.shape[1])
    return avg_error


# grab the image
original_image = np.array(Image.open('../../Data/shopping-street.jpg'))

