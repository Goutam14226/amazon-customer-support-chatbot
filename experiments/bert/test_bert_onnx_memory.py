import os
import psutil
import onnxruntime as ort
from transformers import AutoTokenizer


MODEL_DIR = "bert_onnx"
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")


def memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


print("=" * 60)
print("BERT ONNX MEMORY TEST")
print("=" * 60)

print(f"\nMemory before loading: {memory_mb():.2f} MB")

print("\n1. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
print(f"Memory after tokenizer: {memory_mb():.2f} MB")

print("\n2. Loading ONNX model...")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

print("ONNX model loaded.")
print(f"Memory after ONNX model: {memory_mb():.2f} MB")

print("\n3. Running inference...")

query = "My refund has not been processed yet."

inputs = tokenizer(
    query,
    return_tensors="np",
    padding=True,
    truncation=True,
    max_length=128
)

outputs = session.run(
    None,
    {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }
)

print("Inference completed.")
print(f"Memory after inference: {memory_mb():.2f} MB")

print("\n" + "=" * 60)
print("MEMORY TEST COMPLETE")
print("=" * 60)
