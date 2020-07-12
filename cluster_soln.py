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

# set centroids to random pixels
def init_centroids(centroids, img):
    first = (0,0)
    # img.shape = (height, width, 3)
    # img.shape[:-1] = (height, width)
    last = img.shape[:-1]
    for i,(x,y) in enumerate(np.random.randint(first, last, (NUM,2))):
        centroids[i,:] = img[x,y,:]

# get number of closest centroid to pxl
def closest_pxl(centroids, pxl):
    return np.argmin([np.linalg.norm(pxl-centroids[k]) for k in range(NUM)])

# train centroid model
def train(centroids, img):
    # new image with each pixel represented by a centroid number
    closest = np.zeros(img[:].shape[:-1])

    while True:
        # approximate each pixel by closest centroid
        for x in range(img[:].shape[0]):
            for y in range(img[:].shape[1]):
                closest[x,y] = closest_pxl(centroids, img[x,y,:])

        # update centroids to be average pixel in centroid cluster
        new_centroids = np.copy(centroids)
        for i in range(NUM):
            new_centroids[i,:] = np.average(img[closest==i], axis=0)

        # check for convergence
        print(np.linalg.norm(centroids-new_centroids))
        if np.linalg.norm(centroids-new_centroids) < 5:
            break

        # process update
        centroids[:] = new_centroids[:]

init_centroids(centroids, img)

train(centroids, img)

# replace image pixels with value of closest centroid
for x in range(img[:].shape[0]):
    for y in range(img[:].shape[1]):
        img[x,y,:] = centroids[closest_pxl(centroids, img[x,y,:]),:]

imageio.imsave('newface.png', img.astype('uint8'))