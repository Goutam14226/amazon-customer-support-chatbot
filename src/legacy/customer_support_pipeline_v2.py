import os
import re

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from transformers import pipeline


# ============================================================
# CONFIG
# ============================================================

INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

MIN_SIMILARITY = 0.35

BERT_MODEL = "SandipanMondal06/bert-routing-agent"


# ============================================================
# STOPWORDS
# ============================================================

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


# ============================================================
# INITIALIZE
# ============================================================

print("Connecting to Pinecone...")

pc = Pinecone(
    api_key=os.environ["PINECONE_API_KEY"]
)

index = pc.Index(INDEX_NAME)

print("Pinecone connected.")


print("\nLoading embedding model...")

embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


print("\nConnecting to LLaMA...")

llm = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
    provider="auto"
)

print("LLaMA connection ready.")


# BERT will be loaded only when escalation is required.
bert_router = None


# ============================================================
# UTILITY
# ============================================================

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


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(query):

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
        return [], 0.0, 0.0

    scores = [
        match["score"]
        for match in matches
    ]

    top_score = scores[0]

    avg_score = sum(scores) / len(scores)

    return matches, top_score, avg_score


# ============================================================
# DOMAIN GUARD
# ============================================================

def domain_guard(query, matches, top_score, avg_score):

    if not matches:
        return False, ["no retrieval results"]

    query_words = meaningful_words(query)

    retrieved_text = " ".join(
        match.get("metadata", {}).get("text", "")
        for match in matches
    )

    context_words = meaningful_words(
        retrieved_text
    )

    overlap = query_words.intersection(
        context_words
    )

    overlap_ratio = (
        len(overlap) / len(query_words)
        if query_words
        else 0.0
    )

    reasons = []

    if top_score < MIN_SIMILARITY:
        reasons.append(
            "low retrieval similarity"
        )

    if avg_score < MIN_SIMILARITY:
        reasons.append(
            "low average similarity"
        )

    if overlap_ratio == 0:
        reasons.append(
            "no meaningful query-term overlap"
        )

    supported = (
        top_score >= MIN_SIMILARITY
        and avg_score >= MIN_SIMILARITY
        and overlap_ratio > 0
    )

    return supported, reasons


# ============================================================
# BERT ROUTER
# ============================================================

def route_department(query):

    global bert_router

    if bert_router is None:

        print("\nLoading BERT routing model...")

        bert_router = pipeline(
            "text-classification",
            model=BERT_MODEL
        )

        print("BERT routing model loaded.")

    result = bert_router(
        query,
        top_k=1
    )[0]

    return result["label"], result["score"]


# ============================================================
# LLAMA GENERATION
# ============================================================

def generate_answer(query, matches):

    contexts = []

    for match in matches:

        metadata = match.get(
            "metadata",
            {}
        )

        text = metadata.get(
            "text",
            ""
        )

        if text:
            contexts.append(text)

    context = "\n\n".join(
        contexts
    )

    prompt = f"""
You are an Amazon customer support assistant.

Use ONLY the policy context provided below.

Your job is to answer the customer's question
using the retrieved policy information.

IMPORTANT:

1. Do not use outside knowledge.
2. Do not invent policies.
3. Do not invent procedures.
4. Do not invent refund timelines.
5. Do not invent fees or guarantees.
6. If the policy context does NOT contain enough
   information to answer the customer's question,
   return exactly:

INSUFFICIENT_CONTEXT

7. If the context contains enough information,
   return the actual concise customer-support answer.
8. Do not mention this instruction.

Customer Question:
{query}

Policy Context:
{context}
"""

    try:

        response = llm.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict "
                        "policy-grounded "
                        "customer support assistant."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=200,
            temperature=0.0
        )

        answer = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        return answer

    except Exception as e:

        error_text = str(e)

        if "402" in error_text:
            return (
                "HF_CREDITS_EXHAUSTED"
            )

        return (
            "LLAMA_ERROR"
        )


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_query(query):

    print("\n" + "=" * 80)

    print("USER QUERY:")
    print(query)

    # --------------------------------------------------------
    # 1. RETRIEVAL
    # --------------------------------------------------------

    matches, top_score, avg_score = (
        retrieve_documents(query)
    )

    print("\nRetrieval Confidence:")
    print(round(top_score, 6))

    print("Average Similarity:")
    print(round(avg_score, 6))

    # --------------------------------------------------------
    # 2. DOMAIN GUARD
    # --------------------------------------------------------

    supported, reasons = domain_guard(
        query,
        matches,
        top_score,
        avg_score
    )

    if not supported:

        print("\nDomain Guard:")
        print("ESCALATION")

        if reasons:
            print("\nReasons:")

            for reason in reasons:
                print("-", reason)

        department, routing_score = (
            route_department(query)
        )

        print("\nDepartment:")
        print(department)

        print("Routing Confidence:")
        print(round(routing_score, 6))

        return {
            "status": "ESCALATED",
            "department": department,
            "routing_score": routing_score,
            "retrieval_score": top_score
        }

    # --------------------------------------------------------
    # 3. GENERATION
    # --------------------------------------------------------

    print("\nDomain Guard:")
    print("CANDIDATE FOR GENERATION")

    print("\nGenerating LLaMA response...")

    answer = generate_answer(
        query,
        matches
    )

    # --------------------------------------------------------
    # 4. HANDLE LLaMA FAILURE
    # --------------------------------------------------------

    if answer == "HF_CREDITS_EXHAUSTED":

        print("\nLLaMA:")
        print("HF credits exhausted.")

        print("\nDecision:")
        print("ESCALATION")

        department, routing_score = (
            route_department(query)
        )

        print("\nDepartment:")
        print(department)

        print("Routing Confidence:")
        print(round(routing_score, 6))

        return {
            "status": "ESCALATED",
            "department": department,
            "routing_score": routing_score,
            "reason": "HF credits exhausted"
        }

    if answer == "LLAMA_ERROR":

        print("\nLLaMA:")
        print("Generation error.")

        print("\nDecision:")
        print("ESCALATION")

        department, routing_score = (
            route_department(query)
        )

        print("\nDepartment:")
        print(department)

        print("Routing Confidence:")
        print(round(routing_score, 6))

        return {
            "status": "ESCALATED",
            "department": department,
            "routing_score": routing_score,
            "reason": "LLaMA generation error"
        }

    # --------------------------------------------------------
    # 5. CHECK INSUFFICIENT CONTEXT
    # --------------------------------------------------------

    if (
        "INSUFFICIENT_CONTEXT"
        in answer.upper()
    ):

        print("\nLLaMA:")
        print("Insufficient context.")

        print("\nDecision:")
        print("ESCALATION")

        department, routing_score = (
            route_department(query)
        )

        print("\nDepartment:")
        print(department)

        print("Routing Confidence:")
        print(round(routing_score, 6))

        return {
            "status": "ESCALATED",
            "department": department,
            "routing_score": routing_score,
            "reason": "Insufficient policy context"
        }

    # --------------------------------------------------------
    # 6. FINAL RESPONSE
    # --------------------------------------------------------

    print("\nLLaMA ANSWER:")
    print(answer)

    print("\nDecision:")
    print("RESPOND")

    return {
        "status": "RESPOND",
        "answer": answer,
        "retrieval_score": top_score
    }


# ============================================================
# INTERACTIVE MODE
# ============================================================

if __name__ == "__main__":

    print("\nCustomer Support Pipeline")
    print("Type 'exit' to stop.")

    while True:

        query = input(
            "\nCustomer: "
        ).strip()

        if query.lower() == "exit":
            print("Exiting...")
            break

        if not query:
            continue

        process_query(query)
