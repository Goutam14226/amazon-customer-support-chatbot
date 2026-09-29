import os
import numpy as np

from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer


MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT = "Where is my refund?"


print("=== Local embedding ===")

local_model = SentenceTransformer(MODEL)

local_embedding = local_model.encode(
    TEXT,
    normalize_embeddings=True
)

print("Local dimension:", len(local_embedding))


print("\n=== Hugging Face hosted embedding ===")

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
    provider="hf-inference"
)

hosted_embedding = client.feature_extraction(
    TEXT,
    model=MODEL,
    normalize=True
)

hosted_embedding = np.asarray(
    hosted_embedding,
    dtype=np.float32
)

print("Hosted shape:", hosted_embedding.shape)

hosted_embedding = hosted_embedding.reshape(-1)

print("Hosted dimension:", len(hosted_embedding))


print("\n=== Comparison ===")

cosine_similarity = float(
    np.dot(local_embedding, hosted_embedding)
)

print("Cosine similarity:", cosine_similarity)

if len(hosted_embedding) == 384:
    print("SUCCESS: Hosted embedding is 384-D.")
else:
    print("ERROR: Hosted embedding dimension is not 384.")
