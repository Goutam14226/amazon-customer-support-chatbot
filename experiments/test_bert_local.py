import time
import psutil
import os

from transformers import pipeline


MODEL = "SandipanMondal06/bert-routing-agent"

process = psutil.Process(os.getpid())


def memory_mb():
    return process.memory_info().rss / (1024 * 1024)


print("=" * 60)
print("LOCAL BERT MEMORY TEST")
print("=" * 60)

print(f"Memory before loading: {memory_mb():.2f} MB")

start = time.time()

print("\nLoading BERT routing model...")

router = pipeline(
    "text-classification",
    model=MODEL,
    top_k=1
)

load_time = time.time() - start

print("BERT model loaded.")

print(f"Memory after loading: {memory_mb():.2f} MB")
print(f"Memory used by process: {memory_mb():.2f} MB")
print(f"Loading time: {load_time:.2f} seconds")


print("\n" + "=" * 60)
print("ROUTING TEST")
print("=" * 60)


test_queries = [
    "My refund has not been processed yet.",
    "I want to cancel my order.",
    "I did not receive my package.",
    "I have a problem with my payment.",
    "I want to update my account details.",
]


for query in test_queries:

    result = router(query, top_k=1)[0]

    print("\nQuery:", query)
    print("Department:", result["label"])
    print("Confidence:", round(result["score"], 6))


print("\n" + "=" * 60)
print("FINAL MEMORY")
print("=" * 60)

print(f"Final process memory: {memory_mb():.2f} MB")