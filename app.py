import os
import re
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pinecone import Pinecone
from huggingface_hub import InferenceClient


# ============================================================
# CONFIG
# ============================================================

INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

MIN_SIMILARITY = 0.35


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Amazon Customer Support Chatbot",
    version="1.0.0"
)


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
# EXPLICIT HUMAN ESCALATION PHRASES
# ============================================================

ESCALATION_PHRASES = [
    "human help",
    "human support",
    "human agent",
    "support agent",
    "customer support agent",
    "talk to an agent",
    "speak to an agent",
    "talk to someone",
    "speak to someone",
    "talk to a human",
    "speak to a human",
    "human representative",
    "customer representative",
    "customer service representative",
    "live agent",
    "live support",
    "connect me to support",
    "connect me to an agent",
    "connect me with an agent",
    "connect me to a human",
    "i need an agent",
    "i need human help",
    "i need human support",
    "i want an agent",
    "i want human support",
    "i want to talk to someone",
    "i want to speak to someone",
    "i want to talk to an agent",
    "i want to speak to an agent",
    "please escalate",
    "escalate this",
    "escalate my issue",
    "transfer me",
    "transfer me to support",
]


# ============================================================
# INITIALIZATION
# ============================================================

print("Connecting to Pinecone...")

pc = Pinecone(
    api_key=os.environ["PINECONE_API_KEY"]
)

index = pc.Index(INDEX_NAME)

print("Pinecone connected.")

print("Connecting to LLaMA...")

llm = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
    provider="auto"
)

print("LLaMA connection ready.")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embedding(text: str):
    embedding = llm.feature_extraction(
        text,
        model=EMBEDDING_MODEL,
        normalize=True
    )

    return np.asarray(embedding, dtype=np.float32).reshape(-1)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    query: str


# ============================================================
# EXPLICIT ESCALATION CHECK
# ============================================================

def wants_human_support(query):

    text = query.lower().strip()

    for phrase in ESCALATION_PHRASES:

        if phrase in text:
            return True

    return False


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

    query_embedding = get_embedding(query).tolist()

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
            "reasons": ["no retrieval results"]
        }

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
# LIGHTWEIGHT DEPARTMENT ROUTING
# ============================================================

ROUTING_RULES = {

    "REFUND": [
        "refund",
        "refunded",
        "money back",
        "money hasn't come back",
        "money has not come back",
        "haven't received my money",
        "have not received my money",
        "refund not received",
        "refund not processed",
        "refund status",
        "where is my refund",
    ],

    "RETURN": [
        "return",
        "returning",
        "returned item",
        "send back",
        "send the item back",
        "return an item",
        "return this item",
    ],

    "ORDER": [
        "cancel my order",
        "cancel order",
        "cancel the order",
        "order cancellation",
        "modify my order",
        "change my order",
    ],

    "DELIVERY": [
        "delivery",
        "deliver",
        "package",
        "parcel",
        "not received",
        "not delivered",
        "hasn't arrived",
        "has not arrived",
        "didn't arrive",
        "did not arrive",
        "where is my package",
        "where is my parcel",
        "shipping",
    ],

    "PAYMENT": [
        "payment",
        "payment problem",
        "payment issue",
        "problem with my payment",
        "failed payment",
        "payment failed",
        "unable to pay",
        "can't pay",
        "cannot pay",
        "charged",
        "billing",
    ],

    "ACCOUNT": [
        "account",
        "account details",
        "account information",
        "profile",
        "personal details",
        "update my details",
        "update account",
        "change my account",
    ],
}


def route_department(query):

    q = query.lower().strip()

    scores = {}

    for department, phrases in ROUTING_RULES.items():

        score = 0

        for phrase in phrases:

            if phrase in q:
                score += 1

        scores[department] = score

    best_department = max(
        scores,
        key=scores.get
    )

    best_score = scores[best_department]

    if best_score == 0:
        return "CONTACT", 0.0

    return best_department, float(best_score)

# ============================================================
# ESCALATION
# ============================================================

def escalate(query, reason):

    print("Escalation Reason:", reason)

    department, routing_score = route_department(
        query
    )

    return {
        "status": "ESCALATED",
        "department": department,
        "routing_score": routing_score,
        "reason": reason
    }


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

    context = "\n\n".join(contexts)

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
   information to answer the question,
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

        print("LLaMA Error:", str(e))

        if "402" in str(e):
            return "HF_CREDITS_EXHAUSTED"

        return "LLAMA_ERROR"





# ============================================================
# RESPONSE SUFFICIENCY CHECK
# ============================================================

INSUFFICIENT_RESPONSE_PHRASES = [
    "insufficient_context",
    "insufficient context",
    "i need more information",
    "need more information",
    "need additional information",
    "more information is needed",
    "more details are needed",
    "please provide more information",
    "please provide more details",
    "unable to assist",
    "unable to help",
    "cannot assist",
    "cannot help",
    "can't assist",
    "can't help",
    "not enough information",
    "not enough details",
    "do not have enough information",
    "don't have enough information",
    "does not contain enough information",
]


def response_is_insufficient(answer):

    if not answer:
        return True

    text = answer.lower().strip()

    for phrase in INSUFFICIENT_RESPONSE_PHRASES:

        if phrase in text:
            return True

    return False


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_query(query):

    query = query.strip()

    if not query:

        return {
            "status": "ERROR",
            "message": "Query cannot be empty."
        }

    # --------------------------------------------------------
    # STEP 1 — EXPLICIT HUMAN REQUEST
    # --------------------------------------------------------

    if wants_human_support(query):

        return escalate(
            query,
            "Customer explicitly requested human support"
        )

    # --------------------------------------------------------
    # STEP 2 — RETRIEVAL
    # --------------------------------------------------------

    matches, top_score, avg_score = (
        retrieve_documents(query)
    )

    # --------------------------------------------------------
    # STEP 3 — DOMAIN GUARD
    # --------------------------------------------------------

    guard = domain_guard(
        query,
        matches,
        top_score,
        avg_score
    )

    if not guard["is_support_query"]:

        return {
            "status": "OUT_OF_DOMAIN",
            "message": (
                "I'm designed to assist with Amazon "
                "customer-support queries. Please ask "
                "me about orders, returns, refunds, "
                "payments, delivery, or related Amazon "
                "customer-support issues."
            ),
            "retrieval_score": top_score
        }

    # --------------------------------------------------------
    # STEP 4 — LLAMA
    # --------------------------------------------------------

    answer = generate_answer(
        query,
        matches
    )

    # --------------------------------------------------------
    # STEP 5 — LLAMA FAILURE
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
    # STEP 6 — RESPONSE SUFFICIENCY
    # --------------------------------------------------------

    if response_is_insufficient(answer):

        return escalate(
            query,
            "LLaMA indicated that the available information was insufficient"
        )

    # --------------------------------------------------------
    # STEP 7 — RESPOND
    # --------------------------------------------------------

    return {
        "status": "RESPOND",
        "answer": answer,
        "retrieval_score": top_score
    }


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Amazon Customer Support Chatbot API",
        "status": "running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    try:

        result = process_query(
            request.query
        )

        return result

    except Exception as e:

        print("API Error:", str(e))

        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )