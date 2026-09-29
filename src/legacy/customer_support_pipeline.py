import os

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from transformers import pipeline


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

# Provisional threshold based on our calibration tests
CONFIDENCE_THRESHOLD = 0.35

LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BERT_MODEL = "SandipanMondal06/bert-routing-agent"


# ============================================================
# INITIALIZE PINECONE
# ============================================================

print("Connecting to Pinecone...")

pc = Pinecone(
    api_key=os.environ["PINECONE_API_KEY"]
)

index = pc.Index(INDEX_NAME)

print("Pinecone connected.")


# ============================================================
# INITIALIZE EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# INITIALIZE LLAMA
# ============================================================

print("\nConnecting to LLaMA...")

llm = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
    provider="auto"
)

print("LLaMA connection ready.")


# ============================================================
# INITIALIZE BERT ROUTER
# ============================================================

print("\nLoading BERT routing model...")

router = pipeline(
    "text-classification",
    model=BERT_MODEL
)

print("BERT routing model loaded.")


# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve_documents(query, top_k=3):

    query_embedding = embed_model.encode(query).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    matches = results["matches"]

    max_score = max(
        (match["score"] for match in matches),
        default=0.0
    )

    contexts = []

    for match in matches:

        metadata = match.get("metadata", {})

        text = metadata.get("text", "")

        if text:
            contexts.append({
                "text": text,
                "score": match["score"],
                "source": metadata.get("source", "Unknown"),
                "page": metadata.get("page", "Unknown"),
                "chunk": metadata.get("chunk", "Unknown")
            })

    return max_score, contexts


# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_answer(query, contexts):

    context_text = "\n\n".join(
        item["text"]
        for item in contexts
    )

    prompt = f"""
You are an Amazon customer support assistant.

Answer the customer's question using ONLY the policy
context provided below.

If the policy context does not contain enough information
to answer the question, say that the available policy
information is insufficient.

Do not invent:
- policies
- refund timelines
- guarantees
- procedures
- fees
- delivery promises

Customer Question:
{query}

Policy Context:
{context_text}

Provide a clear and concise customer-support answer.
"""

    response = llm.chat.completions.create(
        model=LLAMA_MODEL,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful and "
                    "policy-grounded customer support assistant."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        max_tokens=200,
        temperature=0.2
    )

    return response.choices[0].message.content


# ============================================================
# DEPARTMENT ROUTING
# ============================================================

def route_department(query):

    result = router(query)

    department = result[0]["label"]
    confidence = result[0]["score"]

    return department, confidence


# ============================================================
# MAIN CUSTOMER SUPPORT PIPELINE
# ============================================================

def process_query(query):

    print("\n" + "=" * 70)
    print("CUSTOMER QUERY:")
    print(query)

    # --------------------------------------------------------
    # Step 1: Retrieval
    # --------------------------------------------------------

    retrieval_score, contexts = retrieve_documents(query)

    print("\nRetrieval Confidence:")
    print(round(retrieval_score, 6))

    # --------------------------------------------------------
    # Step 2: Confidence Decision
    # --------------------------------------------------------

    if retrieval_score >= CONFIDENCE_THRESHOLD:

        print("\nDecision: GENERATION PATH")

        # ----------------------------------------------------
        # Step 3: Generate answer using LLaMA
        # ----------------------------------------------------

        answer = generate_answer(
            query,
            contexts
        )

        print("\nLLaMA RESPONSE:")
        print(answer)

        return {
            "decision": "generation",
            "retrieval_score": retrieval_score,
            "answer": answer
        }

    # --------------------------------------------------------
    # Low confidence → Escalation
    # --------------------------------------------------------

    else:

        print("\nDecision: ESCALATION PATH")

        # ----------------------------------------------------
        # Domain / Support Check
        # ----------------------------------------------------

        print("\nQuery has low retrieval confidence.")

        # For now, we route low-confidence queries
        # to the department classifier.

        department, bert_confidence = route_department(query)

        print("\nBERT ROUTING:")
        print("Department:", department)
        print("BERT Confidence:", round(bert_confidence, 6))

        print("\nHuman Support Escalation:")
        print(
            f"Your query has been escalated to the "
            f"{department} support department."
        )

        return {
            "decision": "escalation",
            "retrieval_score": retrieval_score,
            "department": department,
            "bert_confidence": bert_confidence
        }


# ============================================================
# INTERACTIVE CHAT
# ============================================================

print("\n" + "=" * 70)
print("CUSTOMER SUPPORT CHATBOT")
print("=" * 70)

print("\nType 'exit' to stop.\n")


while True:

    query = input("You: ").strip()

    if query.lower() == "exit":
        print("\nChatbot stopped.")
        break

    if not query:
        continue

    try:

        process_query(query)

    except Exception as e:

        print("\nERROR:")
        print(e)
