import os

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient


# =========================
# Configuration
# =========================

PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
HF_TOKEN = os.environ["HF_TOKEN"]


# =========================
# Initialize Pinecone
# =========================

pc = Pinecone(
    api_key=os.environ["PINECONE_API_KEY"]
)

index = pc.Index(PINECONE_INDEX_NAME)


# =========================
# Initialize Embedding Model
# =========================

embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# =========================
# Initialize LLaMA
# =========================

llm = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto"
)


# =========================
# User Query
# =========================

query = "Where is my refund?"


# =========================
# Query Embedding
# =========================

query_embedding = embed_model.encode(
    query
).tolist()


# =========================
# Pinecone Retrieval
# =========================

results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True
)


# =========================
# Display Retrieved Context
# =========================

print("\n==============================")
print("RETRIEVED DOCUMENTS")
print("==============================\n")

contexts = []

for i, match in enumerate(results["matches"], start=1):

    score = match["score"]
    metadata = match.get("metadata", {})

    print(f"--- Document {i} ---")
    print("Score:", score)
    print("Source:", metadata.get("source"))
    print("Page:", metadata.get("page"))
    print("Chunk:", metadata.get("chunk_id"))
    print()

    text = metadata.get("text", "")

    print(text[:1000])
    print()

    contexts.append(text)


# =========================
# Combine Context
# =========================

context = "\n\n".join(contexts)


# =========================
# RAG Prompt
# =========================

prompt = f"""
You are an Amazon customer support assistant.

Answer the customer's question using ONLY the policy context provided below.

If the policy context does not contain enough information to answer the question,
say that the available policy information is insufficient.

Do not invent policies, refund timelines, guarantees, or procedures.

Customer Question:
{query}

Policy Context:
{context}

Provide a clear and concise customer-support answer.
"""


# =========================
# LLaMA Generation
# =========================

response = llm.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful and policy-grounded customer support assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    max_tokens=200,
    temperature=0.2
)


# =========================
# Final Answer
# =========================

print("\n==============================")
print("LLAMA RAG RESPONSE")
print("==============================\n")

print(response.choices[0].message.content)
