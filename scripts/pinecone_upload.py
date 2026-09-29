import json
import os
from getpass import getpass
from pathlib import Path

from pinecone import Pinecone

INDEX_NAME = "amazon-policies"
EXPECTED_DIMENSION = 384
BATCH_SIZE = 100

# This script uploads the already-generated vectors from embeddings.json.
# It does NOT regenerate embeddings, so it does not require PyTorch/SentenceTransformers.

def find_embeddings_file():
    candidates = [
        Path("embeddings.json"),
        Path("Multi-Agent-LLM-Customer-Support-System-with-Retrieval-Augmented-Resolution-Escalation-main/embeddings.json"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "embeddings.json not found. Put this script in the project folder "
        "or run it from the folder containing embeddings.json."
    )

def main():
    embeddings_path = find_embeddings_file()
    print(f"Reading: {embeddings_path}")

    with embeddings_path.open("r", encoding="utf-8") as f:
        vectors = json.load(f)

    if not isinstance(vectors, list) or not vectors:
        raise ValueError("embeddings.json does not contain a non-empty list.")

    print(f"Found {len(vectors)} vectors.")

    for i, v in enumerate(vectors):
        if "id" not in v or "values" not in v:
            raise ValueError(f"Vector {i} is missing 'id' or 'values'.")
        if len(v["values"]) != EXPECTED_DIMENSION:
            raise ValueError(
                f"Vector {v['id']} has dimension {len(v['values'])}; "
                f"expected {EXPECTED_DIMENSION}."
            )

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        api_key = getpass("Enter your Pinecone API key (input stays hidden): ").strip()

    if not api_key:
        raise ValueError("Pinecone API key was empty.")

    pc = Pinecone(api_key=api_key)

    index_names = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in index_names:
        raise RuntimeError(
            f"Index '{INDEX_NAME}' was not found. Create it first in Pinecone."
        )

    index = pc.Index(INDEX_NAME)
    print(f"Connected to Pinecone index: {INDEX_NAME}")

    print("Uploading vectors...")
    total = len(vectors)

    for start in range(0, total, BATCH_SIZE):
        batch = vectors[start:start + BATCH_SIZE]
        index.upsert(vectors=batch)
        end = min(start + BATCH_SIZE, total)
        print(f"Uploaded {end}/{total}")

    print("\nUpload request completed.")
    print("Checking index statistics...")
    stats = index.describe_index_stats()

    print(f"Total vector count reported by Pinecone: {stats.total_vector_count}")

    if stats.total_vector_count >= total:
        print("\nSUCCESS: Pinecone contains all uploaded vectors.")
    else:
        print(
            "\nWARNING: Pinecone count is lower than expected. "
            "Wait a few seconds and run this script again to verify."
        )

if __name__ == "__main__":
    main()
