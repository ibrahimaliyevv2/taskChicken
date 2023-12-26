# import os
# from PIL import Image
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# import cv2


# import torch
# import clip

# # Load the OpenAI CLIP model with only the image encoder
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device)
# model.visual.eval()

# # Replace these with the paths to your video files
# video_paths = ["./videos/10 Second Video Loop.mp4", "./videos/10 Seconds Of Chickens.mp4"]

# # Parameters for clustering and PCA
# num_clusters = 2  # Adjust as needed
# pca_components = 2

# # Initialize a list to store video embeddings
# video_embeddings = []

# # Process each video and obtain video embeddings
# for video_path in video_paths:
#     print(f"Processing video: {video_path}")

#     # Capture the video
#     cap = cv2.VideoCapture(video_path)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     # Initialize a list to store frame embeddings
#     frame_embeddings = []

#     # Process each frame and obtain image embeddings
#     for _ in range(frame_count):
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert the frame to RGB and preprocess for the CLIP model
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Convert NumPy array to PIL image
#         image_pil = Image.fromarray(image)

#         image_input = preprocess(image_pil).unsqueeze(0).to(device)

#         # Obtain image embedding using the CLIP model
#         with torch.no_grad():
#             image_embedding = model.encode_image(image_input)

#         frame_embeddings.append(image_embedding.cpu().numpy()[0])  # Use [0] to get the actual embedding

#     cap.release()

#     # Calculate the average embedding for the video
#     average_video_embedding = np.mean(frame_embeddings, axis=0)
#     video_embeddings.append(average_video_embedding)

# # Convert embeddings to numpy array
# video_embeddings = np.array(video_embeddings)

# # Apply KMeans clustering on the video embeddings
# kmeans = KMeans(n_clusters=num_clusters, n_init=10)
# video_cluster_assignments = kmeans.fit_predict(video_embeddings)

# # Apply PCA to reduce dimensionality of cluster centers to the specified number of components
# pca = PCA(n_components=pca_components)
# cluster_centers_2d = pca.fit_transform(kmeans.cluster_centers_)

# # Visualize the cluster centers in 2D
# import matplotlib.pyplot as plt

# plt.figure(figsize=(8, 8))
# for cluster_id in range(num_clusters):
#     plt.scatter(cluster_centers_2d[cluster_id, 0], cluster_centers_2d[cluster_id, 1], label=f"Cluster {cluster_id + 1}")

# plt.title("Video Clustering Visualization using KMeans and PCA")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.legend()
# plt.show()


import os
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import cv2

import torch
import clip

# Load the OpenAI CLIP model with only the image encoder
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)
model.visual.eval()

# Replace these with the paths to your video files
video_paths = ["./videos/10 Second Video Loop.mp4", "./videos/10 Seconds Of Chickens.mp4"]

# Parameters for clustering and PCA
num_clusters = 2  # Adjust as needed
pca_components = 2

# Initialize a list to store video embeddings
video_embeddings = []

# Process each video and obtain video embeddings
for video_path in video_paths:
    print(f"Processing video: {video_path}")

    # Capture the video
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize a list to store frame embeddings
    frame_embeddings = []

    # Process each frame and obtain image embeddings
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB and preprocess for the CLIP model
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL image
        image_pil = Image.fromarray(image)

        image_input = preprocess(image_pil).unsqueeze(0).to(device)

        # Obtain image embedding using the CLIP model
        with torch.no_grad():
            image_embedding = model.encode_image(image_input)

        # Use [0] to get the actual embedding
        frame_embeddings.append(image_embedding.cpu().numpy()[0])

    cap.release()

    # Calculate the average embedding for the video
    average_video_embedding = np.mean(frame_embeddings, axis=0)
    video_embeddings.append(average_video_embedding)

# Convert embeddings to numpy array
video_embeddings = np.array(video_embeddings)

# Print the number of features returned by CLIP model
print(f"Number of features returned by CLIP model: {video_embeddings.shape[1]}")

# Apply KMeans clustering on the video embeddings
kmeans = KMeans(n_clusters=num_clusters, n_init=10)
video_cluster_assignments = kmeans.fit_predict(video_embeddings)

# Print the number of features KMeans clustering over
print(f"Number of features KMeans clustering over: {video_embeddings.shape[1]}")

# Apply PCA to reduce dimensionality of cluster centers to the specified number of components
pca = PCA(n_components=pca_components)
video_embeddings_2d = pca.fit_transform(video_embeddings)

# Visualize the cluster centers in 2D
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
for cluster_id in range(num_clusters):
    plt.scatter(video_embeddings_2d[video_cluster_assignments == cluster_id, 0],
                video_embeddings_2d[video_cluster_assignments == cluster_id, 1],
                label=f"Cluster {cluster_id + 1}")

plt.title("Video Clustering Visualization using KMeans and PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()
