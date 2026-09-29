import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# =========================
# Configuration
# =========================

INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
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
# BERT Department Router
# =========================

print("\nLoading BERT routing model...")

router = pipeline(
    "text-classification",
    model="SandipanMondal06/bert-routing-agent"
)

print("BERT routing model loaded.\n")

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
# Test Pipeline
# =========================

for query in queries:

    print("=" * 70)
    print("Query:", query)

    # 1. Create embedding
    query_embedding = embed_model.encode(query).tolist()

    # 2. Retrieve from Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    matches = results["matches"]

    # 3. Get highest retrieval similarity
    max_score = max(
        (match["score"] for match in matches),
        default=0.0
    )

    print("Retrieval Confidence:", round(max_score, 6))

    # 4. Confidence decision
    if max_score >= CONFIDENCE_THRESHOLD:

        print("Decision: GENERATION PATH")
        print("→ Query is sufficiently relevant to retrieved policy.")

    else:

        print("Decision: ESCALATION PATH")

        # 5. BERT Department Routing
        route = router(query)

        department = route[0]["label"]
        bert_confidence = route[0]["score"]

        print("BERT Department:", department)
        print("BERT Confidence:", round(bert_confidence, 6))

    print()
