import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import VAE

# Use the GPU if available
# Check for cuda on windows or linux systems, and use the GPU
# If it is macos, check for MPS and use the GPU
if os.name == "nt":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
elif os.name == "posix":
    if os.uname().sysname == "Darwin":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Using {device}")

# Define the latent space size for the VAE
latent_size = 20  # Can be tuned

# Read in the data file, select ROI (keeping all features), and extract dataset shape and number of features
data = np.load("./imgs/EBSD_data.npy")[:, :448, :512]

# Put the features in the last dimension
data = np.moveaxis(data, 0, -1)
"""
Structure of the data: (n_features, n_rows, n_columns)
Features: Euler_1, Euler_2, Euler_3, GROD, IPF001_1, IPF001_2, IPF001_3, IPF100_1, IPF100_2, IPF100_3, MisIPF001_1, MisIPF001_2, MisIPF001_3, MisIPF100_1, MisIPF100_2, MisIPF100_3, KAM, CI, Phase, IQ
"""

height, width, n_features = data.shape
batch_size = 1 # Because we are using a single image

# Preprocess data to be between 0 and 1 and have a mean of 0 and standard deviation of 1
input_data = (data - data.min(axis=(1, 2))[:, None, None]) / (data.max(axis=(1, 2)) - data.min(axis=(1, 2)))[:, None, None]
input_data = (input_data - input_data.mean(axis=(1, 2))[:, None, None]) / input_data.std(axis=(1, 2))[:, None, None]
input_data = torch.from_numpy(input_data).float().unsqueeze(0).to(device)
input_data = torch.moveaxis(input_data, -1, 1)  # (B, C, H, W)

# Example usage:
vae = VAE.VAutoencoder(input_data.shape[2], input_data.shape[3], n_features, latent_size).to(device)
vae.train(input_data, 100)

# Sample a few latent space vectors (analogous to PCA vectors)
vae_data = vae.get_latent_vectors(input_data, num_samples=3).cpu().numpy()
vae_data = np.squeeze(vae_data)

# Process the results into images
new_data = data.dot(vae_data.T)
grod = data[:, :, 3]
ci = data[:, :, -1]
ipf = data[:, :, 4:7].astype(int)

# Visualize
fig, ax = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
ax[0, 0].imshow(new_data[:, :, 0], cmap="gray")
ax[0, 1].imshow(new_data[:, :, 1], cmap="gray")
ax[0, 2].imshow(new_data[:, :, 2], cmap="gray")
ax[1, 0].imshow(grod, cmap="gray")
ax[1, 1].imshow(ci, cmap="gray")
ax[1, 2].imshow(ipf)
for a in ax.ravel():
    a.axis("off")
labels = ["VAE1", "VAE2", "VAE3", "GROD", "CI", "IPF"]
for i, a in enumerate(ax.ravel()):
    a.text(0.02, 0.98, labels[i], color="black", transform=a.transAxes, fontsize=14, verticalalignment="top", fontname="Avenir", fontweight="heavy", backgroundcolor="white")

plt.tight_layout()
plt.show()
