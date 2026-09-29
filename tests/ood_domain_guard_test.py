import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# =========================
# Configuration
# =========================

INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

# Current provisional threshold
CONFIDENCE_THRESHOLD = 0.35

# =========================
# Pinecone
# =========================

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
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
    # Customer-support queries
    "Where is my refund?",
    "I want to cancel my order.",
    "I did not receive my package.",
    "How can I update my account details?",
    "I have a problem with my payment.",
    "How do I return an item?",
    "How can I track my order?",

    # Clearly unrelated queries
    "What is the weather today?",
    "How do I configure my Wi-Fi router?",
    "Who won the football match?",
    "Give me a recipe for pasta.",
    "What is the capital of France?"
]

# =========================
# Retrieval + Domain Guard
# =========================

for query in queries:

    print("=" * 70)
    print("Query:", query)

    # Create embedding
    query_embedding = embed_model.encode(query).tolist()

    # Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    matches = results["matches"]

    # Highest similarity
    max_score = max(
        (match["score"] for match in matches),
        default=0.0
    )

    print("Retrieval Confidence:", round(max_score, 6))

    # =========================
    # Domain Guard
    # =========================

    if max_score >= CONFIDENCE_THRESHOLD:

        print("Domain Decision: CUSTOMER SUPPORT")
        print("Next Step: GENERATION")

    else:

        print("Domain Decision: LOW CONFIDENCE")
        print("Next Step: ESCALATION / DOMAIN CHECK")

    print()
