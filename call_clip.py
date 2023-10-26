import numpy as np
from sklearn.cluster import KMeans
import torch
from PIL import Image
import clip

# Load the OpenAI CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# Generate image embeddings using only the image encoder of CLIP
image_paths = ["./content/photoimagenew.jpg", "./content/photonext.jpg"]
image_embeddings = []
for path in image_paths:
    image = preprocess(Image.open(path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = model.encode_image(image)
    image_embeddings.append(image_embedding.cpu().numpy())

# Normalize the embeddings
normalized_embeddings = np.array(image_embeddings) / np.linalg.norm(image_embeddings, axis=1, keepdims=True)

# Reshape the normalized_embeddings array
normalized_embeddings = normalized_embeddings.reshape(len(image_paths), -1)

# Apply KMeans clustering
num_clusters = 2  # Adjust as needed
kmeans = KMeans(n_clusters=num_clusters, n_init=10)
cluster_assignments = kmeans.fit_predict(normalized_embeddings)

# Now cluster_assignments contains the cluster IDs for each image
print(cluster_assignments)