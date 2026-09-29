import os
import re

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

# Provisional thresholds based on our previous tests
MIN_SIMILARITY = 0.35
MIN_AVG_SIMILARITY = 0.30
MIN_TOP_GAP = 0.005


print("Connecting to Pinecone...")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
print("Pinecone connected.")

print("\nLoading embedding model...")
embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model loaded.")


queries = [
    "Where is my refund?",
    "I want to cancel my order.",
    "I did not receive my package.",
    "How can I update my account details?",
    "I have a problem with my payment.",
    "How do I return an item?",
    "How can I track my order?",
    "What is the weather today?",
    "How do I configure my Wi-Fi router?",
    "Who is the president of France?",
    "How do I cook pasta?",
]


def normalize(text):
    return set(
        re.findall(
            r"\b[a-zA-Z]{3,}\b",
            text.lower()
        )
    )


def analyze_query(query):

    query_embedding = embed_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    matches = results["matches"]

    scores = [
        match["score"]
        for match in matches
    ]

    if not scores:
        return {
            "decision": "ESCALATE",
            "reason": "No retrieval results"
        }

    top_score = scores[0]

    second_score = scores[1] if len(scores) > 1 else 0.0

    avg_score = sum(scores) / len(scores)

    gap = top_score - second_score

    # ------------------------------------------------
    # Lexical overlap
    # ------------------------------------------------

    query_words = normalize(query)

    retrieved_text = " ".join(
        match.get("metadata", {}).get("text", "")
        for match in matches
    )

    context_words = normalize(retrieved_text)

    overlap = query_words.intersection(context_words)

    overlap_ratio = (
        len(overlap) / len(query_words)
        if query_words
        else 0
    )

    # ------------------------------------------------
    # Decision
    # ------------------------------------------------

    reasons = []

    if top_score < MIN_SIMILARITY:
        reasons.append("low top similarity")

    if avg_score < MIN_AVG_SIMILARITY:
        reasons.append("low average similarity")

    if gap < MIN_TOP_GAP:
        reasons.append("very small top-1/top-2 gap")

    if overlap_ratio == 0:
        reasons.append("no lexical overlap")

    if reasons:
        decision = "ESCALATE"
    else:
        decision = "CANDIDATE_FOR_GENERATION"

    return {
        "decision": decision,
        "top_score": top_score,
        "second_score": second_score,
        "avg_score": avg_score,
        "gap": gap,
        "overlap_ratio": overlap_ratio,
        "overlap_words": sorted(overlap),
        "reasons": reasons,
        "matches": matches,
    }


for query in queries:

    print("\n" + "=" * 90)
    print("QUERY:")
    print(query)

    result = analyze_query(query)

    print("\nTop Similarity:")
    print(round(result.get("top_score", 0), 6))

    print("Second Similarity:")
    print(round(result.get("second_score", 0), 6))

    print("Average Similarity:")
    print(round(result.get("avg_score", 0), 6))

    print("Top-1 / Top-2 Gap:")
    print(round(result.get("gap", 0), 6))

    print("Lexical Overlap:")
    print(round(result.get("overlap_ratio", 0), 4))

    print("Overlap Words:")
    print(result.get("overlap_words", []))

    print("\nDecision:")
    print(result["decision"])

    if result.get("reasons"):
        print("\nReasons:")
        for reason in result["reasons"]:
            print("-", reason)

    print("\nTop Sources:")

    for i, match in enumerate(
        result.get("matches", []),
        start=1
    ):
        metadata = match.get("metadata", {})

        print(
            f"{i}. "
            f"{round(match['score'], 6)} | "
            f"{metadata.get('source', 'Unknown')}"
        )