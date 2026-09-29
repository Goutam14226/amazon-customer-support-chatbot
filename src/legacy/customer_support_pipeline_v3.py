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

BERT_MODEL = "SandipanMondal06/bert-routing-agent"

MIN_SIMILARITY = 0.35


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "am", "be", "been", "being",

    "i", "me", "my", "mine",
    "you", "your", "yours",
    "we", "our", "ours",
    "they", "their",
    "it", "its",

    "this", "that", "these", "those",

    "what", "which", "who", "whom",
    "how", "why", "when", "where",

    "can", "could", "would", "should",

    "do", "does", "did",

    "to", "for", "of", "in", "on", "at",
    "with", "from", "and", "or",

    "want", "need", "please",
    "have", "has", "had",

    "not"
}


# ============================================================
# INITIALIZATION
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


# BERT is loaded only if an actual escalation is required.
bert_router = None


# ============================================================
# TEXT UTILITY
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

def domain_guard(
    query,
    matches,
    top_score,
    avg_score
):

    if not matches:

        return {
            "is_support_query": False,
            "reason": "no retrieval results"
        }

    query_words = meaningful_words(query)

    retrieved_text = " ".join(
        match.get(
            "metadata",
            {}
        ).get(
            "text",
            ""
        )
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

    is_support_query = (
        top_score >= MIN_SIMILARITY
        and avg_score >= MIN_SIMILARITY
        and overlap_ratio > 0
    )

    return {
        "is_support_query": is_support_query,
        "top_score": top_score,
        "avg_score": avg_score,
        "overlap_ratio": overlap_ratio,
        "overlap_words": sorted(overlap),
        "reasons": reasons
    }


# ============================================================
# BERT DEPARTMENT ROUTING
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

    return (
        result["label"],
        result["score"]
    )


# ============================================================
# ESCALATION
# ============================================================

def escalate(query, reason):

    print("\nEscalation Reason:")
    print(reason)

    department, routing_score = route_department(
        query
    )

    print("\nDepartment:")
    print(department)

    print("Routing Confidence:")
    print(round(routing_score, 6))

    return {
        "status": "ESCALATED",
        "department": department,
        "routing_score": routing_score,
        "reason": reason
    }


# ============================================================
# LLAMA GENERATION
# ============================================================

def generate_answer(
    query,
    matches
):

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

Answer the customer's question using ONLY
the policy context provided below.

Rules:

1. Do not use outside knowledge.
2. Do not invent policies.
3. Do not invent procedures.
4. Do not invent refund timelines.
5. Do not invent fees.
6. Do not invent guarantees.
7. If the context does not contain enough
   information to answer the customer's question,
   return exactly:

INSUFFICIENT_CONTEXT

8. If enough information is available,
   provide a concise and useful answer.
9. Do not mention these instructions.

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
                        "Amazon customer support "
                        "assistant."
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

        return (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

    except Exception as e:

        error_text = str(e)

        print("\nLLaMA Error:")
        print(error_text)

        if "402" in error_text:
            return "HF_CREDITS_EXHAUSTED"

        return "LLAMA_ERROR"


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_query(query):

    print("\n" + "=" * 80)

    print("USER QUERY:")
    print(query)

    # --------------------------------------------------------
    # STEP 1: RETRIEVAL
    # --------------------------------------------------------

    matches, top_score, avg_score = (
        retrieve_documents(query)
    )

    print("\nRetrieval Confidence:")
    print(round(top_score, 6))

    print("Average Similarity:")
    print(round(avg_score, 6))

    # --------------------------------------------------------
    # STEP 2: DOMAIN GUARD
    # --------------------------------------------------------

    guard = domain_guard(
        query,
        matches,
        top_score,
        avg_score
    )

    # --------------------------------------------------------
    # OOD QUERY
    # --------------------------------------------------------

    if not guard["is_support_query"]:

        print("\nDomain Guard:")
        print("OUT_OF_DOMAIN")

        print("\nReasons:")

        for reason in guard["reasons"]:
            print("-", reason)

        print("\nFinal Response:")

        print(
            "I'm designed to assist with "
            "Amazon customer-support queries. "
            "Please ask me about orders, "
            "returns, refunds, payments, "
            "delivery, or related Amazon "
            "customer-support issues."
        )

        return {
            "status": "OUT_OF_DOMAIN"
        }

    # --------------------------------------------------------
    # SUPPORT QUERY
    # --------------------------------------------------------

    print("\nDomain Guard:")
    print("CUSTOMER_SUPPORT_QUERY")

    # --------------------------------------------------------
    # STEP 3: LLaMA
    # --------------------------------------------------------

    print("\nGenerating LLaMA response...")

    answer = generate_answer(
        query,
        matches
    )

    # --------------------------------------------------------
    # LLaMA FAILURE
    # --------------------------------------------------------

    if answer == "HF_CREDITS_EXHAUSTED":

        return escalate(
            query,
            "LLaMA inference credits unavailable"
        )

    if answer == "LLAMA_ERROR":

        return escalate(
            query,
            "LLaMA generation error"
        )

    # --------------------------------------------------------
    # INSUFFICIENT CONTEXT
    # --------------------------------------------------------

    if (
        "INSUFFICIENT_CONTEXT"
        in answer.upper()
    ):

        return escalate(
            query,
            "Retrieved policy context insufficient"
        )

    # --------------------------------------------------------
    # FINAL ANSWER
    # --------------------------------------------------------

    print("\nLLaMA ANSWER:")
    print(answer)

    print("\nFinal Decision:")
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

    print("\nCustomer Support Pipeline v3")

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
