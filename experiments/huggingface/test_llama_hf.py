import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
    provider="auto"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful customer support assistant."
        },
        {
            "role": "user",
            "content": "Where is my refund?"
        }
    ],
    max_tokens=100,
    temperature=0.2
)

print("\n===== LLaMA RESPONSE =====\n")
print(response.choices[0].message.content)
