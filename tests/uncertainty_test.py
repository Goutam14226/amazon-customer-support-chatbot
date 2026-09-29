import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=os.environ["HF_TOKEN"]
)

print("Tokenizer loaded.")

print("\nLoading LLaMA model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=os.environ["HF_TOKEN"],
    torch_dtype=torch.float32,
    device_map="auto"
)

model.eval()

print("LLaMA model loaded.")


# ============================================================
# TEST QUERY
# ============================================================

query = "Where is my refund?"

prompt = f"""
You are an Amazon customer support assistant.

Answer this customer question clearly using the available
customer-support information.

Customer Question:
{query}

Answer:
"""


# ============================================================
# TOKENIZE
# ============================================================

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

inputs = {
    key: value.to(model.device)
    for key, value in inputs.items()
}


# ============================================================
# GENERATE
# ============================================================

print("\nGenerating response...")

with torch.no_grad():

    output = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.2,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True
    )


# ============================================================
# DECODE
# ============================================================

generated_tokens = output.sequences[
    0,
    inputs["input_ids"].shape[1]:
]

response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)

print("\nLLaMA Response:")
print(response)


# ============================================================
# PERPLEXITY
# ============================================================

scores = output.scores

log_probs = []

for step, logits in enumerate(scores):

    # Probability distribution
    log_probabilities = torch.log_softmax(
        logits[0],
        dim=-1
    )

    # Token generated at this step
    token_id = generated_tokens[step]

    token_log_probability = log_probabilities[
        token_id
    ]

    log_probs.append(
        token_log_probability.item()
    )


if log_probs:

    average_negative_log_likelihood = (
        -sum(log_probs) / len(log_probs)
    )

    perplexity = torch.exp(
        torch.tensor(average_negative_log_likelihood)
    ).item()

else:

    perplexity = float("inf")


print("\nPerplexity:")
print(round(perplexity, 6))


# ============================================================
# ENTROPY
# ============================================================

entropies = []

for logits in scores:

    probabilities = torch.softmax(
        logits[0],
        dim=-1
    )

    entropy = -torch.sum(
        probabilities * torch.log(
            probabilities + 1e-12
        )
    )

    entropies.append(
        entropy.item()
    )


if entropies:

    average_entropy = sum(entropies) / len(entropies)

else:

    average_entropy = float("inf")


print("\nAverage Entropy:")
print(round(average_entropy, 6))
