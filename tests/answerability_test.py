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
    "What is the weather today?",
]


# ============================================================
# ANSWERABILITY CHECK
# ============================================================

def check_answerability(query, context):

    prompt = f"""
You are evaluating whether a customer-support policy context
contains enough information to answer a customer's question.

Customer Question:
{query}

Policy Context:
{context}

Determine whether the policy context contains enough
information to provide a useful and grounded answer.

Return ONLY one of these two labels:

ANSWERABLE

or

NOT_ANSWERABLE

Do not provide an explanation.
Do not answer the customer's question.
"""


    response = llm.chat.completions.create(
        model=LLAMA_MODEL,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict context "
                    "answerability evaluator."
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

    # Clean possible formatting
    result = re.sub(
        r"[^A-Za-z_]",
        "",
        result
    ).upper()

    if "NOTANSWERABLE" in result:
        return "NOT_ANSWERABLE"

    if "ANSWERABLE" in result:
        return "ANSWERABLE"

    return "UNKNOWN"


# ============================================================
# MAIN TEST
# ============================================================

for query in queries:

    print("\n" + "=" * 80)

    print("QUERY:")
    print(query)

    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query_embedding = embed_model.encode(
        query
    ).tolist()

    # --------------------------------------------------------
    # Pinecone retrieval
    # --------------------------------------------------------

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
    # Very low retrieval confidence
    # --------------------------------------------------------

    if max_score < CONFIDENCE_THRESHOLD:

        print("\nInitial Decision:")
        print("ESCALATION")

        print(
            "\nAnswerability Check skipped "
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
    # Answerability
    # --------------------------------------------------------

    print("\nRunning answerability check...")

    answerability = check_answerability(
        query,
        context
    )

    print("\nAnswerability:")
    print(answerability)


    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    if answerability == "ANSWERABLE":

        print("\nFinal Decision:")
        print("GENERATION")

    elif answerability == "NOT_ANSWERABLE":

        print("\nFinal Decision:")
        print("ESCALATION")

    else:

        print("\nFinal Decision:")
        print("ESCALATION")

    print()
