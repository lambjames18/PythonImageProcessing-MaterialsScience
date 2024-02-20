import numpy as np
from skimage import io, filters, exposure
import matplotlib.pyplot as plt


def norm(image):
    return np.around(255 * (image - image.min(axis=(0, 1))) / (image.max(axis=(0, 1)) - image.min(axis=(0, 1)))).astype(np.uint8)


def un_norm(image):
    image = image.astype(float)
    return 2 * ((image - image.min(axis=(0, 1))) / (image.max(axis=(0, 1)) - image.min(axis=(0, 1)))) - 1


# Read in the image
im = io.imread("./imgs/EBSD-Pattern.tif").astype(float)

# Create a dynamic background (really blurred version of the image) and subtract it from the original image
background = filters.gaussian(im, sigma=50)
im_new = un_norm(im - background)

# Apply contrast limiting adaptive histogram equalization (CLAHE) to the background-removed image
im_new_clahe = exposure.equalize_adapthist(im_new, clip_limit=0.01)

# Normalize the images and save them
im = norm(im)
background = norm(background)
im_new = norm(im_new)
im_new_clahe = norm(im_new_clahe)

io.imsave("./imgs/EBSD-Pattern-Background.tif", background)
io.imsave("./imgs/EBSD-Pattern-Background-Removed.tif", im_new)
io.imsave("./imgs/EBSD-Pattern-Background-Removed-CLAHE.tif", im_new_clahe)
