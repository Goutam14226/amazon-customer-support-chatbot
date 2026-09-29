import os
import re

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

CONFIDENCE_THRESHOLD = 0.35

LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


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
]


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(query, context):

    prompt = f"""
You are an Amazon customer support assistant.

Answer the customer's question using ONLY the policy
context provided below.

Do not use outside knowledge.

Do not invent:
- policies
- procedures
- refund timelines
- guarantees
- fees
- delivery promises

If the context does not contain enough information,
say that the available policy information is insufficient.

Customer Question:
{query}

Policy Context:
{context}

Provide a concise answer.
"""

    response = llm.chat.completions.create(
        model=LLAMA_MODEL,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict, "
                    "policy-grounded customer support assistant."
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

    return response.choices[0].message.content.strip()


# ============================================================
# GROUNDING CHECK
# ============================================================

def check_grounding(query, answer, context):

    prompt = f"""
You are a strict evaluator of a customer-support answer.

Customer Question:
{query}

Policy Context:
{context}

Generated Answer:
{answer}

Evaluate whether every important factual claim in the
generated answer is supported by the provided policy context.

Use these rules:

1. If the answer contains factual claims that are supported
   by the policy context, consider it grounded.

2. If the answer invents policies, procedures, timelines,
   guarantees, fees, or other facts not present in the
   context, consider it not grounded.

3. Do not use outside knowledge.

Return ONLY one of:

GROUNDED

NOT_GROUNDED

Do not provide an explanation.
"""

    response = llm.chat.completions.create(
        model=LLAMA_MODEL,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict factual "
                    "grounding evaluator."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        max_tokens=10,
        temperature=0.0
    )

    result = response.choices[0].message.content.strip()

    result = re.sub(
        r"[^A-Za-z_]",
        "",
        result
    ).upper()

    if "NOTGROUNDED" in result:
        return "NOT_GROUNDED"

    if "GROUNDED" in result:
        return "GROUNDED"

    return "UNKNOWN"


# ============================================================
# MAIN TEST
# ============================================================

for query in queries:

    print("\n" + "=" * 80)

    print("QUERY:")
    print(query)

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    query_embedding = embed_model.encode(
        query
    ).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    matches = results["matches"]

    max_score = max(
        (match["score"] for match in matches),
        default=0.0
    )

    print("\nRetrieval Confidence:")
    print(round(max_score, 6))

    # --------------------------------------------------------
    # Low confidence
    # --------------------------------------------------------

    if max_score < CONFIDENCE_THRESHOLD:

        print("\nDecision:")
        print("ESCALATION")

        print(
            "Grounding check skipped "
            "(retrieval confidence too low)."
        )

        continue

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    contexts = []

    for match in matches:

        metadata = match.get("metadata", {})

        text = metadata.get(
            "text",
            ""
        )

        if text:
            contexts.append(text)

    context = "\n\n".join(contexts)

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    print("\nGenerating LLaMA answer...")

    answer = generate_answer(
        query,
        context
    )

    print("\nLLaMA ANSWER:")
    print(answer)

    # --------------------------------------------------------
    # Grounding
    # --------------------------------------------------------

    print("\nChecking grounding...")

    grounding = check_grounding(
        query,
        answer,
        context
    )

    print("\nGrounding Result:")
    print(grounding)

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    if grounding == "GROUNDED":

        print("\nFinal Decision:")
        print("RESPOND")

    else:

        print("\nFinal Decision:")
        print("ESCALATE")

    print()
