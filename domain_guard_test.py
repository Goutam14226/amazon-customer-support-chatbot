import os
import re

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

# Provisional threshold from our previous tests
MIN_SIMILARITY = 0.35

# Common words that should NOT count as meaningful overlap
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "am", "be", "been", "being",
    "i", "me", "my", "mine", "you", "your", "yours",
    "we", "our", "ours", "they", "their",
    "it", "its",
    "this", "that", "these", "those",
    "what", "which", "who", "whom",
    "how", "why", "when", "where",
    "can", "could", "would", "should",
    "do", "does", "did",
    "to", "for", "of", "in", "on", "at",
    "with", "from", "and", "or",
    "want", "need", "please",
    "have", "has", "had"
}


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

    # OOD queries
    "What is the weather today?",
    "How do I configure my Wi-Fi router?",
    "Who is the president of France?",
    "How do I cook pasta?",
]


def meaningful_words(text):
    words = re.findall(
        r"\b[a-zA-Z]{3,}\b",
        text.lower()
    )

    return {
        word
        for word in words
        if word not in STOPWORDS
    }


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

    if not matches:
        return {
            "decision": "ESCALATE",
            "reason": "No retrieval results"
        }

    scores = [
        match["score"]
        for match in matches
    ]

    top_score = scores[0]

    avg_score = sum(scores) / len(scores)

    # ---------------------------------------------
    # Meaningful query words
    # ---------------------------------------------

    query_words = meaningful_words(query)

    # ---------------------------------------------
    # Collect retrieved text
    # ---------------------------------------------

    retrieved_text = " ".join(
        match.get("metadata", {}).get("text", "")
        for match in matches
    )

    context_words = meaningful_words(retrieved_text)

    overlap = query_words.intersection(context_words)

    if query_words:
        overlap_ratio = len(overlap) / len(query_words)
    else:
        overlap_ratio = 0.0

    # ---------------------------------------------
    # Source consistency
    # ---------------------------------------------

    sources = []

    for match in matches:
        source = match.get("metadata", {}).get(
            "source",
            "Unknown"
        )
        sources.append(source)

    unique_sources = len(set(sources))

    # ---------------------------------------------
    # Decision
    # ---------------------------------------------

    reasons = []

    if top_score < MIN_SIMILARITY:
        reasons.append("low retrieval similarity")

    if avg_score < MIN_SIMILARITY:
        reasons.append("low average similarity")

    if overlap_ratio == 0:
        reasons.append("no meaningful query-term overlap")

    if top_score >= MIN_SIMILARITY and overlap_ratio > 0:
        decision = "CANDIDATE_FOR_GENERATION"
    else:
        decision = "ESCALATE"

    return {
        "decision": decision,
        "top_score": top_score,
        "avg_score": avg_score,
        "query_words": sorted(query_words),
        "overlap_words": sorted(overlap),
        "overlap_ratio": overlap_ratio,
        "unique_sources": unique_sources,
        "sources": sources,
        "reasons": reasons,
    }


for query in queries:

    print("\n" + "=" * 90)

    print("QUERY:")
    print(query)

    result = analyze_query(query)

    print("\nTop Similarity:")
    print(round(result.get("top_score", 0), 6))

    print("Average Similarity:")
    print(round(result.get("avg_score", 0), 6))

    print("\nMeaningful Query Words:")
    print(result.get("query_words", []))

    print("\nMeaningful Overlap:")
    print(result.get("overlap_words", []))

    print("\nOverlap Ratio:")
    print(round(result.get("overlap_ratio", 0), 4))

    print("\nUnique Sources:")
    print(result.get("unique_sources", 0))

    print("\nDecision:")
    print(result["decision"])

    if result.get("reasons"):
        print("\nReasons:")
        for reason in result["reasons"]:
            print("-", reason)