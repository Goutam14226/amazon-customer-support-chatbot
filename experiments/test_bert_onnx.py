import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_DIR = "bert_onnx"
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")

print("=" * 60)
print("BERT ONNX ROUTING TEST")
print("=" * 60)

print("\n1. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
print("Tokenizer loaded.")

print("\n2. Loading ONNX model...")
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)
print("ONNX model loaded.")

print("\n3. Model inputs:")
for inp in session.get_inputs():
    print(f"  {inp.name}: {inp.shape} | {inp.type}")

print("\n4. Testing routing...\n")

queries = [
    "My refund has not been processed yet.",
    "I want to cancel my order.",
    "I did not receive my package.",
    "I have a problem with my payment.",
    "How can I update my account details?"
]

# Labels from the original BERT routing model
labels = {
    0: "ACCOUNT",
    1: "CANCEL",
    2: "CONTACT",
    3: "DELIVERY",
    4: "FEEDBACK",
    5: "INVOICE",
    6: "ORDER",
    7: "PAYMENT",
    8: "REFUND",
    9: "SHIPPING",
    10: "SUBSCRIPTION",
}

for query in queries:

    inputs = tokenizer(
        query,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=128
    )

    # ONNX model expects these inputs
    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }

    # token_type_ids may or may not be required
    input_names = [x.name for x in session.get_inputs()]

    if "token_type_ids" in input_names:
        ort_inputs["token_type_ids"] = inputs["token_type_ids"]

    outputs = session.run(None, ort_inputs)

    logits = outputs[0][0]

    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()

    predicted_id = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_id])

    department = labels.get(
        predicted_id,
        f"UNKNOWN_LABEL_{predicted_id}"
    )

    print(f"Query      : {query}")
    print(f"Department : {department}")
    print(f"Confidence : {confidence:.6f}")
    print("-" * 60)

print("\nTEST COMPLETED.")