import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL = "SandipanMondal06/bert-routing-agent"
OUTPUT_DIR = "bert_onnx"

print("=" * 60)
print("BERT → ONNX EXPORT")
print("=" * 60)

print("\n1. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
print("Tokenizer loaded.")

print("\n2. Loading BERT model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()
print("BERT model loaded.")

print("\n3. Creating example inputs...")

text = "My refund has not been processed yet."

inputs = tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128
)

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

print("Input shape:", input_ids.shape)

print("\n4. Exporting to ONNX...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.onnx.export(
    model,
    (input_ids, attention_mask),
    os.path.join(OUTPUT_DIR, "model.onnx"),
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch"}
    },
    opset_version=17
)

tokenizer.save_pretrained(OUTPUT_DIR)

print("\nONNX export completed.")
print("Output folder:", os.path.abspath(OUTPUT_DIR))