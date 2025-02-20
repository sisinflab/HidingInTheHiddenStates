from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np
import torch
import os
import math


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_llm(huggingface_path="facebook/opt-350m"):
    model = AutoModelForCausalLM.from_pretrained(
        huggingface_path, torch_dtype="auto", device_map="auto"
    )  # ("facebook/opt-125m")#("facebook/opt-1.3b")
    tokenizer = AutoTokenizer.from_pretrained(
        huggingface_path
    )  # ("facebook/opt-125m")#("facebook/opt-1.3b")
    model.eval()

    return model, tokenizer


def enable_determinism(random_seed=123):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
        possible_values = [value.lower() for value in possible_values]

    perplexity_scores = {}
    for value in possible_values:
        # Generate input text for the model
        if value in special_cases:
            input_text = special_cases[value]  # Use the fixed sentence
        else:
            input_text = template.format(value)  # Normal sentence construction

        # Tokenize the input text
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

        # Calculate loss using the model
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss)  # Perplexity = exp(loss)

        # Store the perplexity score
        perplexity_scores[value] = perplexity.item()

    # Sort possible values based on their perplexity scores (ascending)
    sorted_perplexities = sorted(perplexity_scores.items(), key=lambda x: x[1])

    # Check if current_val is in the top-5 lowest perplexity scores
    top_5_lowest_perplexity = [
        item[0].lower().strip() for item in sorted_perplexities[:5]
    ]

    if current_val.lower().strip() not in top_5_lowest_perplexity:
        return False

    # Exclude the positive sample (current_val) from negatives
    negatives = [
        value
        for value in possible_values
        if str(value).strip().lower() != str(current_val).strip().lower()
    ]

    # Compute scaled scores using exponential scaling
    # Adjust this value to control scaling sharpness
    scaled_scores = {
        value: math.exp(-perplexity_scores[value] / temperature) for value in negatives
    }

    # Sort negatives by scaled scores (descending) for Top-k filtering
    sorted_negatives = sorted(scaled_scores.items(), key=lambda x: x[1], reverse=True)

    # Restrict to Top-k candidates
    top_k_negatives = sorted_negatives[:k]
    top_k_values = [item[0] for item in top_k_negatives]
    top_k_scaled_scores = {item[0]: item[1] for item in top_k_negatives}

    # Normalize probabilities for top-k negatives
    total_scaled = sum(top_k_scaled_scores.values())
    probabilities = {
        value: top_k_scaled_scores[value] / total_scaled for value in top_k_values
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
    normalized_top_p_probabilities = [
        prob / top_p_total for prob in top_p_probabilities
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
