import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import torch
import clip

# Load the OpenAI CLIP model with only the image encoder
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)
model.visual.eval()

# Replace these paths with the paths to your own images
image_paths = ["./content/photoimagenew.jpg", "./content/photonext.jpg"]

image_embeddings = []

for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():  # Corrected line
        image_embedding = model.encode_image(image_input)

    # Convert the image embedding to a 1D numpy array
    image_embedding = image_embedding.cpu().numpy().flatten()

    image_embeddings.append(image_embedding)

# Convert embeddings to a 2D numpy array
image_embeddings = np.array(image_embeddings)

# Apply PCA to reduce dimensionality to 2
pca = PCA(n_components=2)
image_embeddings_2d = pca.fit_transform(image_embeddings)

# Apply KMeans clustering on the 2D embeddings
num_clusters = 2  # Adjust as needed
kmeans = KMeans(n_clusters=num_clusters, n_init=10)
cluster_assignments = kmeans.fit_predict(image_embeddings_2d)

# Visualize the clusters in 2D
plt.figure(figsize=(8, 8))
for cluster_id in range(num_clusters):
    cluster_indices = np.where(cluster_assignments == cluster_id)[0]
    plt.scatter(image_embeddings_2d[cluster_indices, 0], image_embeddings_2d[cluster_indices, 1], label=f"Cluster {cluster_id + 1}")

plt.title("Image Clustering Visualization using PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()
