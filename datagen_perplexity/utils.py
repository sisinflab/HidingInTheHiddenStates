from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np
import torch
import os
import math


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_llm(huggingface_path="facebook/opt-350m", device=None):

    if device is None:
        device = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        huggingface_path, torch_dtype="auto", device_map=device
    )

    tokenizer = AutoTokenizer.from_pretrained(
        huggingface_path
    )
    model.eval()

    return model, tokenizer


def enable_determinism(random_seed=123, deterministic_algorithms=True):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def perplexity_based_sampling(
    required_param,
    current_val,
    template,
    df,
    model,
    tokenizer,
    k=10,
    p=0.9,
    special_cases=None,
    temperature=1,
    lower=False,
):

    if special_cases is None:
        special_cases = {}

    possible_values = df[required_param].dropna().unique()
    # Preprocess possible values
    possible_values = [str(value).strip() for value in possible_values]
    if lower:
        possible_values = [str(value).lower() for value in possible_values]
        current_val = str(current_val).lower()

    perplexity_scores = {}
    for value in possible_values:
        # Generate input text for the model
        if value in special_cases:
            input_text = special_cases[value]  # Use the fixed sentence
        else:
            input_text = template.format(value.strip())  # Normal sentence construction

        # Tokenize the input text
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

        # Calculate loss using the model
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            #loss = outputs.loss
            #perplexity = torch.exp(loss)
            # Perplexity = exp(loss)

            logits = outputs.logits[:, :-1, :]
            labels = inputs.input_ids[:, 1:]
            log_probs = torch.nn.functional.log_softmax(logits, -1)
            target_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
            mean_log_prob = -target_log_probs.mean()
            perplexity = torch.exp(mean_log_prob)


        # Store the perplexity score
        perplexity_scores[value.strip()] = perplexity.item()

    # Sort possible values based on their perplexity scores (ascending)
    sorted_perplexities = sorted(perplexity_scores.items(), key=lambda x: x[1])

    # Check if current_val is in the top lowest perplexity scores
    top_lowest_perplexity = [
        item[0].strip() for item in sorted_perplexities[:max(int(0.1 * len(possible_values)), 1)]
    ]

    if str(current_val).strip() not in top_lowest_perplexity:
        return False

    positive_perplexity = perplexity_scores[str(current_val).strip()]
    tolerance = 0.1 * positive_perplexity

    # Exclude the positive sample (current_val) from negatives
    negatives = [
        value
        for value in possible_values
        if str(value).strip() != str(current_val).strip() and
           ((perplexity_scores[value.strip()] < positive_perplexity) or abs(perplexity_scores[value.strip()] - positive_perplexity) <= tolerance)  # "higher confusion"
    ]
    if not negatives:
        return False

    min_p = min(perplexity_scores.values())
    max_p = max(perplexity_scores.values())
    normalized_scores = {
        value: (score - min_p) / (max_p - min_p) if max_p > min_p else 0.5
        for value, score in perplexity_scores.items()
    }

    # Step 2: Rescale using exponential transformation
    scaled_scores = {
        value: math.exp(-normalized_scores[value.strip()] / temperature)
        for value in negatives
    }


    # Sort negatives by scaled scores (descending) for Top-k filtering
    sorted_negatives = sorted(scaled_scores.items(), key=lambda x: x[1], reverse=True)

    # Restrict to Top-k candidates
    top_k_negatives = sorted_negatives[:k]
    top_k_values = [item[0] for item in top_k_negatives]
    top_k_scaled_scores = {item[0]: item[1] for item in top_k_negatives}

    # Normalize probabilities for top-k negatives
    total_scaled = sum(top_k_scaled_scores.values())
    if total_scaled == 0:
        print("zero encountered")
        print(top_k_scaled_scores)
    probabilities = {
        value: (top_k_scaled_scores[value] / total_scaled) for value in top_k_values
    }

    # Top-p Sampling (nucleus sampling)
    sorted_values = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    cumulative_prob = 0.0
    top_p_values = []
    top_p_probabilities = []

    # Select the top-p values whose cumulative probability exceeds p
    for value, prob in sorted_values:
        cumulative_prob += prob
        top_p_values.append(value)
        top_p_probabilities.append(prob)
        if cumulative_prob >= p:
            break

    # Normalize the probabilities of the selected top-p values
    top_p_total = sum(top_p_probabilities)
    if top_p_total == 0:
        print("zero encountered")
    normalized_top_p_probabilities = [
        (prob / top_p_total) for prob in top_p_probabilities
    ]

    # Sample a negative value from the top-p set
    sampled_negative = np.random.choice(top_p_values, p=normalized_top_p_probabilities)
    sampled_prob = normalized_top_p_probabilities[top_p_values.index(sampled_negative)]

    # Construct the corresponding sentence
    if sampled_negative in special_cases:
        constructed_sentence = special_cases[sampled_negative]  # Use the fixed sentence
    else:
        constructed_sentence = template.format(
            sampled_negative
        )  # Normal sentence construction

    return constructed_sentence, sampled_prob, perplexity_scores, probabilities
