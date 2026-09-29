import os

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

# Minimum semantic relevance between query and retrieved text
CONTEXT_RELEVANCE_THRESHOLD = 0.35


# ============================================================
# PINECONE
# ============================================================

print("Connecting to Pinecone...")

pc = Pinecone(
    api_key=os.environ["PINECONE_API_KEY"]
)

index = pc.Index(INDEX_NAME)

print("Pinecone connected.")


# ============================================================
# EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# TEST QUERIES
# ============================================================

queries = [
    "Where is my refund?",
    "I want to cancel my order.",
    "I did not receive my package.",
    "How can I update my account details?",
    "I have a problem with my payment.",
    "How do I return an item?",
    "How can I track my order?",
    "What is the weather today?",
]


# ============================================================
# CONTEXT RELEVANCE TEST
# ============================================================

for query in queries:

    print("\n" + "=" * 80)
    print("QUERY:")
    print(query)

    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query_embedding = embed_model.encode(
        query,
        normalize_embeddings=True
    )

    # --------------------------------------------------------
    # Pinecone retrieval
    # --------------------------------------------------------

    results = index.query(
        vector=query_embedding.tolist(),
        top_k=3,
        include_metadata=True
    )

    matches = results["matches"]

    # --------------------------------------------------------
    # Check semantic relevance of each retrieved document
    # --------------------------------------------------------

    relevance_scores = []

    print("\nRetrieved Context Relevance:")

    for i, match in enumerate(matches, start=1):

        metadata = match.get("metadata", {})

        text = metadata.get(
            "text",
            ""
        )

        source = metadata.get(
            "source",
            "Unknown"
        )

        if not text:
            continue

        # Embed retrieved context
        context_embedding = embed_model.encode(
            text,
            normalize_embeddings=True
        )

        # Since embeddings are normalized,
        # dot product = cosine similarity
        relevance = float(
            query_embedding @ context_embedding
        )

        relevance_scores.append(relevance)

        print("\nDocument:", i)
        print("Pinecone Score:",
              round(match["score"], 6))
        print("Context Relevance:",
              round(relevance, 6))
        print("Source:", source)

    # --------------------------------------------------------
    # Overall context relevance
    # --------------------------------------------------------

    max_relevance = max(
        relevance_scores,
        default=0.0
    )

    average_relevance = (
        sum(relevance_scores) / len(relevance_scores)
        if relevance_scores
        else 0.0
    )

    print("\nOverall Context Analysis:")
    print(
        "Maximum Relevance:",
        round(max_relevance, 6)
    )

    print(
        "Average Relevance:",
        round(average_relevance, 6)
    )

    # --------------------------------------------------------
    # Preliminary decision
    # --------------------------------------------------------

    if max_relevance >= CONTEXT_RELEVANCE_THRESHOLD:

        print(
            "\nContext Decision: RELEVANT"
        )

    else:

        print(
            "\nContext Decision: WEAK / IRRELEVANT"
        )