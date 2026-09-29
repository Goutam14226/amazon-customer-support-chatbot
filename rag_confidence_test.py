import os

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


# =========================
# Configuration
# =========================

INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

# Provisional threshold
CONFIDENCE_THRESHOLD = 0.35


# =========================
# Pinecone
# =========================

pc = Pinecone(
    api_key=os.environ["PINECONE_API_KEY"]
)

index = pc.Index(INDEX_NAME)


# =========================
# Embedding Model
# =========================

embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# =========================
# Test Queries
# =========================

queries = [
    "Where is my refund?",
    "I want to cancel my order.",
    "I did not receive my package.",
    "How can I update my account details?",
    "I have a problem with my payment.",
    "How do I return an item?",
    "How can I track my order?",
    "What is the weather today?",
    "How do I configure my Wi-Fi router?"
]


# =========================
# Retrieval + Decision
# =========================

for query in queries:

    query_embedding = embed_model.encode(
        query
    ).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    matches = results["matches"]

    if not matches:
        max_score = 0.0
    else:
        max_score = max(
            match["score"]
            for match in matches
        )

    if max_score >= CONFIDENCE_THRESHOLD:
        decision = "GENERATION PATH"
    else:
        decision = "ESCALATION PATH"

    print("\n" + "=" * 60)
    print("Query:", query)
    print("Max Retrieval Score:", round(max_score, 6))
    print("Decision:", decision)

    if matches:
        print("Top Source:", matches[0].get("metadata", {}).get("source"))