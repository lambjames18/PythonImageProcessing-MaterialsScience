import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, decomposition


# Read in the data file, select ROI (keeping all features), and extract dataset shape and number of features
data = np.load("./imgs/EBSD_data.npy")

# Put the features in the last dimension
data = np.moveaxis(data, 0, -1)

"""
Structure of the data: (n_rows, n_columns, n_features)
Features: Euler_1, Euler_2, Euler_3, GROD, IPF001_1, IPF001_2, IPF001_3, IPF100_1, IPF100_2, IPF100_3, MisIPF001_1, MisIPF001_2, MisIPF001_3, MisIPF100_1, MisIPF100_2, MisIPF100_3, KAM, CI, Phase, IQ
"""

# Get the number of features and the shape of the dataset
n_features = data.shape[-1]
shape = data.shape

# Scale the data and perform PCA
data_scaled = preprocessing.StandardScaler().fit_transform(data.reshape(-1, n_features))
pca = decomposition.PCA(n_components=n_features)
X_pca = pca.fit_transform(data_scaled)
X_pca = X_pca.reshape(shape)

# Plot the results
fig, ax = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
ax[0, 0].imshow(X_pca[:, :, 0], cmap="gray")
ax[0, 1].imshow(X_pca[:, :, 1], cmap="gray")
ax[0, 2].imshow(X_pca[:, :, 2], cmap="gray")
ax[1, 0].imshow(data[:, :, 3], cmap="gray")  # GROD
ax[1, 1].imshow(data[:, :, -1], cmap="gray")  # CI
ax[1, 2].imshow(data[:, :, 4:7].astype(int))  # IPF
for a in ax.ravel():
    a.axis("off")
labels = ["PCA1", "PCA2", "PCA3", "GROD", "CI", "IPF"]
for i, a in enumerate(ax.ravel()):
    a.text(
        0.02,
        0.98,
        labels[i],
        color="black",
        transform=a.transAxes,
        fontsize=14,
        verticalalignment="top",
        fontweight="heavy",
        backgroundcolor="white",
    )

plt.tight_layout()
plt.show()
