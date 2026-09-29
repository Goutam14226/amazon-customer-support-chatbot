from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


MODEL_ID = "SandipanMondal06/bert-routing-agent"

TEST_QUERIES = [
    "Where is my refund?",
    "I want to cancel my order.",
    "I did not receive my package.",
    "How can I update my account details?",
    "I have a problem with my payment.",
]


def predict(model, tokenizer, query):
    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0]
    predicted_id = int(torch.argmax(probabilities))
    confidence = float(probabilities[predicted_id])

    id2label = model.config.id2label
    label = id2label.get(predicted_id, str(predicted_id))

    return label, confidence


def main():
    print("=== BERT DEPARTMENT ROUTING TEST ===")
    print("Model:", MODEL_ID)
    print("\nThe model will be downloaded from Hugging Face if it is not cached.\n")

    print("1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("2. Loading BERT classifier...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()

    print("\n3. Running test queries...\n")

    for i, query in enumerate(TEST_QUERIES, 1):
        label, confidence = predict(model, tokenizer, query)

        print(f"Query {i}: {query}")
        print(f"Predicted department: {label}")
        print(f"Confidence: {confidence:.4f}")
        print("-" * 60)

    print("\n=== BERT ROUTING TEST COMPLETE ===")


if __name__ == "__main__":
    main()
