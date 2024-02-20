import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

import SNOW


def shuffle_labels(labels, seed=0):
    unique_ids = np.unique(labels)[1:]  # Remove 0 because its the background
    np.random.seed(seed)
    np.random.shuffle(unique_ids)
    unique_ids = np.concatenate(([0], unique_ids))  # Add 0 back to the beginning
    shuffled_labels = np.zeros_like(labels)
    for i, unique_id in enumerate(unique_ids):
        shuffled_labels[labels == unique_id] = i
    return shuffled_labels


# Get the image
im = io.imread("./imgs/WCu-Composite.tiff")
tungsten_mask = im > filters.threshold_otsu(im)

# Set the minimum distance between seed points
min_distance = 50

# Calculate the distance to the nearest zero pixel for each pixel of the image
distance = ndi.distance_transform_edt(tungsten_mask)
# Find the peaks in the distance map
coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=tungsten_mask, min_distance=20)
# Create the markers for the watershed algorithm
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
# Label the seed points as markers for the watershed algorithm
markers, _ = ndi.label(mask)
# Perform the watershed algorithm
labels = watershed(-distance, markers, mask=tungsten_mask)

### Now we will use the SNOW algorithm to segment the image
sigma = 0.1
r_max = 5
labels_snow = SNOW.snow(im=tungsten_mask, sigma=sigma, r_max=r_max)

# Randomize the labels to make the plot look better, do this for both the original and SNOW labels
shuffled_labels = shuffle_labels(labels)
shuffled_labels_snow = shuffle_labels(labels_snow)

# Plot the results
fig, ax = plt.subplots(1, 3, figsize=(15, 6), sharex=True, sharey=True)
ax[0].imshow(im, cmap=plt.cm.gray)
ax[0].set_title('Overlapping objects')
ax[0].axis('off')
ax[1].imshow(shuffled_labels, cmap=plt.cm.nipy_spectral)
ax[1].set_title('Separated objects')
ax[1].axis('off')
ax[2].imshow(shuffled_labels_snow, cmap=plt.cm.nipy_spectral)
ax[2].set_title('Separated objects (SNOW)')
ax[2].axis('off')
plt.tight_layout()
plt.show()
