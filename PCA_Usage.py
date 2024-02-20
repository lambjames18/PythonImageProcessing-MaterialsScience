import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, decomposition


# Read in data labels
feature_labels = np.load("./EBSD_labels.npy")

# Read in the data file, select ROI (keeping all features), and extract dataset shape and number of features
data = np.load("./EBSD_data.npy")
n_features = data.shape[0]
shape = data.shape

data_scaled = preprocessing.StandardScaler().fit_transform(data.reshape(n_features, -1).T)

pca = decomposition.PCA(n_components=n_features)
X_pca = pca.fit_transform(data_scaled)
X_pca = X_pca.T.reshape(n_features, *shape[1:])

print(X_pca.shape)

fig, ax = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
ax[0, 0].imshow(X_pca[0], cmap="gray")
ax[0, 1].imshow(X_pca[1], cmap="gray")
ax[0, 2].imshow(X_pca[2], cmap="gray")
ax[1, 0].imshow(data[3], cmap="gray")
ax[1, 1].imshow(data[-1], cmap="gray")
ax[1, 2].imshow(np.moveaxis(data[4:7], 0, -1).astype(int))
for a in ax.ravel():
    a.axis("off")
labels = ["PCA1", "PCA2", "PCA3", "GROD", "CI", "IPF"]
for i, a in enumerate(ax.ravel()):
    a.text(0.02, 0.98, labels[i], color="black", transform=a.transAxes, fontsize=14, verticalalignment="top", fontname="Avenir", fontweight="heavy", backgroundcolor="white")
    
plt.tight_layout()
plt.show()