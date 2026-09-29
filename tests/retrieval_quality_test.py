import os

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

# Current provisional threshold
CONFIDENCE_THRESHOLD = 0.35


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
# RETRIEVAL QUALITY TEST
# ============================================================

for query in queries:

    print("\n" + "=" * 80)
    print("QUERY:")
    print(query)

    # --------------------------------------------------------
    # Create embedding
    # --------------------------------------------------------

    query_embedding = embed_model.encode(
        query
    ).tolist()

    # --------------------------------------------------------
    # Pinecone search
    # --------------------------------------------------------

    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    matches = results["matches"]

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    scores = [
        match["score"]
        for match in matches
    ]

    max_score = max(
        scores,
        default=0.0
    )

    second_score = (
        scores[1]
        if len(scores) > 1
        else 0.0
    )

    score_gap = max_score - second_score

    print("\nRetrieval Analysis:")
    print("Top-1 Score :", round(max_score, 6))
    print("Top-2 Score :", round(second_score, 6))
    print("Score Gap   :", round(score_gap, 6))

    # --------------------------------------------------------
    # Preliminary decision
    # --------------------------------------------------------

    if max_score >= CONFIDENCE_THRESHOLD:
        print("Threshold Decision: GENERATION")
    else:
        print("Threshold Decision: ESCALATION")

    # --------------------------------------------------------
    # Retrieved documents
    # --------------------------------------------------------

    print("\nRetrieved Documents:")

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

        page = metadata.get(
            "page",
            "Unknown"
        )

        chunk = metadata.get(
            "chunk",
            "Unknown"
        )

        print("\n" + "-" * 70)

        print(f"Document #{i}")
        print("Score :", round(match["score"], 6))
        print("Source:", source)
        print("Page  :", page)
        print("Chunk :", chunk)

        print("\nText:")
        print(text[:1000])

    print()
