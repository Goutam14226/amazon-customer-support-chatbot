import json
from getpass import getpass
from pinecone import Pinecone

INDEX_NAME = "amazon-policies"
TOP_K = 5

def main():
    with open("embeddings.json", "r", encoding="utf-8") as f:
        vectors = json.load(f)

    # Use an existing stored vector for the first connectivity/retrieval test.
    # This verifies that the Pinecone index can return relevant stored records
    # without installing a local embedding model yet.
    test_vector = vectors[0]["values"]

    api_key = getpass("Enter your Pinecone API key (input stays hidden): ").strip()
    if not api_key:
        raise ValueError("Pinecone API key was empty.")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    print(f"Querying '{INDEX_NAME}' with an existing 384-D vector...")
    result = index.query(
        vector=test_vector,
        top_k=TOP_K,
        include_metadata=True
    )

    matches = result.get("matches", [])
    print(f"\nReturned {len(matches)} matches:\n")

    for i, match in enumerate(matches, 1):
        metadata = match.get("metadata") or {}
        print(f"--- Match {i} ---")
        print(f"ID: {match.get('id')}")
        print(f"Score: {match.get('score')}")
        print(f"Source: {metadata.get('source')}")
        print(f"Page: {metadata.get('page')}")
        print(f"Chunk ID: {metadata.get('chunk_id')}")
        text = metadata.get("text", "")
        print(f"Text: {str(text)[:500]}")
        print()

    if not matches:
        print("WARNING: Pinecone returned no matches.")
    else:
        print("SUCCESS: Pinecone retrieval is working.")

if __name__ == "__main__":
    main()
