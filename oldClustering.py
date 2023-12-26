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
# image_paths = ["./photos/1.jpg", "./photos/2.png", "./photos/3.jpg", "./photos/5.jpg", "./photos/7.jpg", "./photos/9.jpg", "./photos/10.jpg", "./photos/11.jpg", "./photos/12.jpg"]
image_paths = ["./images/1.jpg", "./images/2.jpg", "./images/3.jpg", "./images/4.jpg", "./images/6.jpg"]
image_embeddings = []

# Print the number of features returned by CLIP model
with torch.no_grad():
    sample_image = Image.open(image_paths[0]).convert("RGB")
    sample_image_input = preprocess(sample_image).unsqueeze(0).to(device)
    sample_embedding = model.encode_image(sample_image_input)
    print(f"Number of features returned by CLIP model: {sample_embedding.shape[1]}")

for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_embedding = model.encode_image(image_input)

    # Convert the image embedding to a 1D numpy array
    image_embedding = image_embedding.cpu().numpy().flatten()

    image_embeddings.append(image_embedding)

# Convert embeddings to a 2D numpy array
image_embeddings = np.array(image_embeddings)

# Apply KMeans clustering on the 2D embeddings
num_clusters = 2  # Adjust as needed
kmeans = KMeans(n_clusters=num_clusters, n_init=10)
cluster_assignments = kmeans.fit_predict(image_embeddings)

# Print the number of features used by KMeans clustering
print(f"Number of features used by KMeans clustering: {image_embeddings.shape[1]}")

# Apply PCA to reduce dimensionality to 2
pca = PCA(n_components=2)
image_embeddings_2d = pca.fit_transform(image_embeddings)

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
