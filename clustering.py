import io
import os
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import torch
import clip

app = FastAPI()

# Load the OpenAI CLIP model with only the image encoder
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)
model.visual.eval()

# Placeholder for image embeddings and KMeans model
image_embeddings = None
kmeans_model = None


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = model.encode_image(image_input)
    return image_embedding.cpu().numpy().flatten()


@app.post("/cluster_images/")
async def cluster_images(files: UploadFile = File(...)):
    global image_embeddings, kmeans_model

    # Process uploaded images and update image_embeddings
    images = await files.read()
    image_embedding = preprocess_image(images)

    if image_embeddings is None:
        image_embeddings = image_embedding.reshape(1, -1)
    else:
        image_embeddings = np.vstack([image_embeddings, image_embedding])

    # Check if clustering needs to be updated
    if kmeans_model is None or image_embeddings.shape[0] > kmeans_model.n_clusters:
        num_clusters = image_embeddings.shape[0]  # Adjust as needed
        kmeans_model = KMeans(n_clusters=num_clusters, n_init=10)
        cluster_assignments = kmeans_model.fit_predict(image_embeddings)

    return JSONResponse(content={"cluster_assignments": cluster_assignments.tolist()})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
