from getpass import getpass

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

INDEX_NAME = "amazon-policies"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

def main():
    query = "Where is my refund?"

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    query_vector = model.encode(query).tolist()
    print(f"Query: {query}")
    print(f"Embedding dimension: {len(query_vector)}")

    api_key = getpass("Enter your Pinecone API key (input stays hidden): ").strip()
    if not api_key:
        raise ValueError("Pinecone API key was empty.")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    print(f"\nSearching Pinecone index: {INDEX_NAME}")
    result = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True
    )

    matches = result.get("matches", [])
    print(f"\nReturned {len(matches)} matches:\n")

    for i, match in enumerate(matches, 1):
        metadata = match.get("metadata") or {}
        print(f"========== Match {i} ==========")
        print(f"Score : {match.get('score')}")
        print(f"Source: {metadata.get('source')}")
        print(f"Page  : {metadata.get('page')}")
        print(f"Chunk : {metadata.get('chunk_id')}")
        print("Text  :")
        print(str(metadata.get("text", ""))[:1200])
        print()

    if matches:
        print("SUCCESS: Real natural-language query retrieval is working.")
    else:
        print("WARNING: No matches returned.")

if __name__ == "__main__":
    main()
