import os
from huggingface_hub import InferenceClient


BERT_MODEL = "SandipanMondal06/bert-routing-agent"


client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
    provider="hf-inference"
)


test_queries = [
    "My refund has not been processed yet.",
    "I want to cancel my order.",
    "I did not receive my package.",
    "I have a problem with my payment.",
    "I want to return an item.",
]


print("=== Hugging Face Hosted BERT Routing Test ===")
print()


for query in test_queries:

    print("Query:", query)

    try:

        result = client.text_classification(
            query,
            model=BERT_MODEL,
            top_k=1
        )

        prediction = result[0]

        print("Department:", prediction.label)
        print("Confidence:", prediction.score)

    except Exception as e:

        print("ERROR:", str(e))

    print("-" * 60)
