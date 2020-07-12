import numpy as np
from scipy import misc
import imageio

# f = misc.face()
# imageio.imsave('face.png', f)

img = imageio.imread('face.png')

# shrink size of img by factor of (k x k) to make algo faster
def blur(img, k):
    new_img = np.zeros(((img.shape[0]-1)//k+1, (img.shape[1]-1)//k+1, 3))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            w = k
            if x//k == new_img[:].shape[0]-1:
                w = (img.shape[0]-1)%k+1
            h = k
            if y//k == new_img[:].shape[1]-1:
                h = (img.shape[1]-1)%k+1
            new_img[x//k,y//k,:] += img[x,y,:]/(w*h)
    return new_img.astype('uint8')

img = blur(img, 8)
# imageio.imsave('faceblur.png', img)

NUM = 16
centroids = np.zeros((NUM,3))

# TODO: set centroids to random pixels
def init_centroids(centroids, img):
    pass

# TODO: get number of closest centroid to pxl
def closest_pxl(centroids, pxl):
    pass

# TODO: train centroid model
def train(centroids, img):
    # new image with each pixel represented by a centroid number
    closest = np.zeros(img[:].shape[:-1])

    while True:
        # approximate each pixel by closest centroid

        # update centroids to be average pixel in centroid cluster

        # check for convergence

        # process update

        break

init_centroids(centroids, img)

train(centroids, img)

# TODO: replace image pixels with value of closest centroid

imageio.imsave('newface.png', img.astype('uint8'))