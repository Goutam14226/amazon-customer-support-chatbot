from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer


MODEL = "SandipanMondal06/bert-routing-agent"
OUTPUT_DIR = Path("bert_onnx")


print("=" * 60)
print("BERT → ONNX CONVERSION")
print("=" * 60)

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("Tokenizer loaded.")

print("\nConverting BERT model to ONNX...")

model = ORTModelForSequenceClassification.from_pretrained(
    MODEL,
    export=True
)

print("ONNX model created.")

OUTPUT_DIR.mkdir(exist_ok=True)

print("\nSaving ONNX model...")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nConversion completed.")
print("Output folder:", OUTPUT_DIR.resolve())