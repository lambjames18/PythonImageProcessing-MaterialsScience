import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Read in the data file, select ROI (keeping all features), and extract dataset shape and number of features
data = np.load("./EBSD_data.npy")[:, :448, :512]
data = np.moveaxis(data, 0, -1)
height, width, n_features = data.shape
batch_size = 1

# Preprocess data to be between 0 and 1 and have a mean of 0 and standard deviation of 1
input_data = (data - data.min(axis=(1, 2))[:, None, None]) / (data.max(axis=(1, 2)) - data.min(axis=(1, 2)))[:, None, None]
input_data = (input_data - input_data.mean(axis=(1, 2))[:, None, None]) / input_data.std(axis=(1, 2))[:, None, None]
input_data = torch.from_numpy(data).float().unsqueeze(0).to(device)
input_data = torch.moveaxis(input_data, -1, 1)  # (B, C, H, W)

# Example usage:
latent_size = 20  # Can be tuned
vae = VAE.VAutoencoder(height, width, n_features, latent_size)
vae.to(device)  # Move the model to the GPU if available

# Define your loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# Training loop
num_epochs = 5000
for epoch in range(num_epochs):
    # Forward pass
    output_data, mu, logvar = vae(input_data)

    # Compute the loss, including the KL divergence term
    reconstruction_loss = criterion(output_data, input_data)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = reconstruction_loss + kl_divergence

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Reconstruction Loss: {reconstruction_loss.item():.4f}, KL Divergence: {kl_divergence.item():.4f}' + " " * 10, end="\r", flush=True)
print(f'Final Loss: {loss.item():.4f}, Final reconstruction Loss: {reconstruction_loss.item():.4f}, KL Divergence: {kl_divergence.item():.4f}' + " " * 20)

# Sample a few latent space vectors (analogous to PCA vectors)
vae_data = vae.get_latent_vectors(vae, input_data[:1], num_samples=3).cpu().numpy()

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
