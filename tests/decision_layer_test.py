import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ============================================================
# Decision Layer Test
# Retrieval -> Similarity Threshold -> Generation / Escalation
# Escalation -> BERT Department Routing
#
# IMPORTANT:
# 0.35 is ONLY a provisional threshold based on the small
# calibration test performed so far. It is NOT a final
# scientifically validated production threshold.
# ============================================================

PINECONE_INDEX_NAME = "amazon-policies"
ROUTER_MODEL = "SandipanMondal06/bert-routing-agent"
THRESHOLD = 0.35

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

def get_pinecone_index():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        api_key = input("Enter your Pinecone API key: ").strip()

    pc = Pinecone(api_key=api_key)
    index = pc.Index(PINECONE_INDEX_NAME)
    return index

def load_router():
    print("\nLoading BERT department router...")
    tokenizer = AutoTokenizer.from_pretrained(ROUTER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(ROUTER_MODEL)
    model.eval()
    return tokenizer, model

def route_department(query, tokenizer, model):
    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)
    confidence, predicted_id = torch.max(probs, dim=-1)

    predicted_id = predicted_id.item()
    confidence = confidence.item()

    label = model.config.id2label.get(
        predicted_id,
        str(predicted_id)
    )

    return label, confidence

def retrieve(query, embedder, index):
    query_vector = embedder.encode(query).tolist()

    result = index.query(
        vector=query_vector,
        top_k=5,
        include_metadata=True
    )

    matches = result.get("matches", [])

    if not matches:
        return 0.0, []

    max_score = max(float(m["score"]) for m in matches)
    return max_score, matches

def print_result(query, max_score, matches, route=None):
    print("\n" + "=" * 72)
    print("QUERY:", query)
    print(f"MAX RETRIEVAL SIMILARITY: {max_score:.6f}")
    print(f"PROVISIONAL THRESHOLD:     {THRESHOLD:.2f}")

    if max_score >= THRESHOLD:
        print("\nDECISION: GENERATION PATH")
        print("Reason: retrieval similarity >= threshold")
        print("Next step: send retrieved context + query to the LLM.")
    else:
        print("\nDECISION: ESCALATION PATH")
        print("Reason: retrieval similarity < threshold")

        if route:
            department, confidence = route
            print(f"BERT DEPARTMENT: {department}")
            print(f"BERT CONFIDENCE:  {confidence:.4f}")

    print("\nTOP RETRIEVED SOURCES:")
    for i, match in enumerate(matches[:5], 1):
        metadata = match.get("metadata") or {}
        source = metadata.get("source") or metadata.get("filename") or "Unknown"
        page = metadata.get("page", "?")
        chunk = metadata.get("chunk_id", metadata.get("chunk", "?"))
        print(
            f"{i}. score={float(match['score']):.6f} | "
            f"page={page} | chunk={chunk} | source={source}"
        )

def main():
    print("=" * 72)
    print("CUSTOMER SUPPORT CHATBOT - DECISION LAYER TEST")
    print("=" * 72)
    print("Flow:")
    print("Query -> MiniLM -> Pinecone -> Similarity Check")
    print("High similarity -> Generation Path")
    print("Low similarity  -> Escalation -> BERT Department Routing")
    print(f"\nUsing PROVISIONAL threshold = {THRESHOLD}")
    print("This threshold still needs larger calibration before production.")

    print("\nLoading MiniLM embedding model...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    index = get_pinecone_index()
    tokenizer, router_model = load_router()

    print("\nRunning test queries...")

    for query in TEST_QUERIES:
        max_score, matches = retrieve(query, embedder, index)

        route = None
        if max_score < THRESHOLD:
            route = route_department(query, tokenizer, router_model)

        print_result(query, max_score, matches, route)

    print("\n" + "=" * 72)
    print("TEST COMPLETE")
    print("=" * 72)
    print("Note:")
    print("- Generation Path means retrieval is considered sufficient for now.")
    print("- Escalation Path means the query is sent to department routing.")
    print("- BERT can still assign a department to unrelated/OOD queries;")
    print("  that does NOT prove the query belongs to that department.")
    print("- Next step after this test: integrate the decision layer into")
    print("  the actual chatbot pipeline and then handle LLM generation.")

if __name__ == "__main__":
    main()
