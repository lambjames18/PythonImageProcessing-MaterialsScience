import numpy as np
from skimage import io, exposure, filters
import torch
import matplotlib.pyplot as plt

import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image = io.imread("./imgs/CeO2.tiff", as_gray=True) - io.imread("./imgs/CeO2_Background.tiff", as_gray=True)
# image[image]
image = image[:, :256]
image = (image - image.min()) / (image.max() - image.min())

# Rotate image
rot_image = np.rot90(image, 1)

# Intensity adjustment
adj_image = exposure.adjust_gamma(image, 0.5)

# Sharpen image
sharpened_image = filters.unsharp_mask(image, radius=1, amount=10)

# Create synthetic data
input_data = (image - image.mean()) / image.std()
input_data = torch.from_numpy(input_data).float().unsqueeze(0).unsqueeze(0).to(device)
vae = VAE.VAutoencoder(input_data.shape[2], input_data.shape[3], 1, 10).to(device)
vae.train(input_data, 5000)
vae_image = vae(input_data)[0].detach().squeeze().cpu().numpy()

# Export the images
image = np.around(255 * (image - image.min()) / (image.max() - image.min())).astype(np.uint8)
rot_image = np.around(255 * (rot_image - rot_image.min()) / (rot_image.max() - rot_image.min())).astype(np.uint8)
adj_image = np.around(255 * (adj_image - adj_image.min()) / (adj_image.max() - adj_image.min())).astype(np.uint8)
sharpened_image = np.around(255 * (sharpened_image - sharpened_image.min()) / (sharpened_image.max() - sharpened_image.min())).astype(np.uint8)
vae_image = np.around(255 * (vae_image - vae_image.min()) / (vae_image.max() - vae_image.min())).astype(np.uint8)

io.imsave("./imgs/CeO2_Original.tiff", image)
io.imsave("./imgs/CeO2_Rotated.tiff", rot_image)
io.imsave("./imgs/CeO2_Adjusted.tiff", adj_image)
io.imsave("./imgs/CeO2_Sharpened.tiff", sharpened_image)
io.imsave("./imgs/CeO2_VAE.tiff", vae_image)
