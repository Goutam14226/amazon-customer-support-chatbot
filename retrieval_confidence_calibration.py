from getpass import getpass

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone


PINECONE_INDEX = "amazon-policies"

TEST_QUERIES = [
    "Where is my refund?",
    "I want to cancel my order.",
    "I did not receive my package.",
    "How can I update my account details?",
    "I have a problem with my payment.",
    "How do I return an item?",
    "How can I track my order?",
    "What is the weather forecast for tomorrow?",
    "How do I reset my Wi-Fi router?",
]


def main():
    print("=== RETRIEVAL CONFIDENCE CALIBRATION TEST ===")
    print("This test reports Pinecone similarity scores.")
    print("It does NOT treat the score as a probability.\n")

    api_key = getpass("Enter your Pinecone API key: ").strip()

    if not api_key:
        print("ERROR: Pinecone API key is required.")
        return

    print("\n1. Loading MiniLM...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("2. Connecting to Pinecone...")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(PINECONE_INDEX)

    print("\n3. Testing retrieval scores...\n")

    for i, query in enumerate(TEST_QUERIES, 1):
        vector = embedder.encode(query).tolist()

        result = index.query(
            vector=vector,
            top_k=5,
            include_metadata=True
        )

        matches = result.get("matches", [])

        if not matches:
            print(f"{i}. {query}")
            print("   No matches returned.")
            print("-" * 70)
            continue

        max_score = matches[0].get("score", 0)

        print(f"{i}. {query}")
        print(f"   Max similarity: {max_score:.6f}")

        for rank, match in enumerate(matches[:3], 1):
            md = match.get("metadata", {})
            print(
                f"   Top {rank}: score={match.get('score', 0):.6f} | "
                f"source={md.get('source', 'Unknown')} | "
                f"page={md.get('page', 'Unknown')}"
            )

        print("-" * 70)

    print("\n=== CALIBRATION COMPLETE ===")
    print("Use these observed scores to choose/calibrate the escalation threshold.")
    print("Do not blindly use a threshold from the project report; the report defines")
    print("a threshold concept (tau_s), but does not provide a validated numeric value.")


if __name__ == "__main__":
    main()
