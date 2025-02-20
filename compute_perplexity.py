from datagen_perplexity.utils import get_llm
from datagen_perplexity.utils import enable_determinism
import importlib
import pandas as pd
import os
import torch



LLMS = [
    "meta-llama/Llama-2-7b-hf",
    "facebook/opt-6.7b",
#    "meta-llama/Meta-Llama-3-8B",
]


DATASETS = [
    "animals",
    "capitals",
    "companies",
    "elements",
    "inventions"
]


def compute_perplexity(input_text, model, tokenizer):

    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    # Calculate loss using the model
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

        logits = outputs.logits[:, :-1, :]
        labels = inputs.input_ids[:, 1:]
        log_probs = torch.nn.functional.log_softmax(logits, -1)
        target_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        mean_log_prob = -target_log_probs.mean()
        perplexity = torch.exp(mean_log_prob).item()

    return perplexity


def compute_dataframe_perplexity(df, model, tokenizer):
    true_pp = []
    false_pp = []
    for _, row in df.iterrows():
        pp = compute_perplexity(row["statement"], model, tokenizer)
        if row["label"] == 1:
            true_pp.append(pp)
        else:
            false_pp.append(pp)

    return true_pp, false_pp


if __name__ == "__main__":
    for llm in LLMS:
        model, tokenizer = get_llm(llm)
        for dataset in DATASETS:
            enable_determinism()
            print(f"Processing dataset: {dataset}")
            # load csv
            path_llm = os.path.join("resources", llm.split("/")[-1], f"{dataset}_true_false.csv")
            path_original = os.path.join("resources", f"{dataset}_true_false.csv")

            df_original = pd.read_csv(path_original)  # Ensure pandas is imported
            df_llm = pd.read_csv(path_llm)  # Ensure pandas is imported
            
            print(len(df_original))
            print(len(df_llm))
            true_pp_original, false_pp_original = compute_dataframe_perplexity(df_original, model, tokenizer)
            true_pp_llm, false_pp_llm = compute_dataframe_perplexity(df_llm, model, tokenizer)


            true_perplexity_original = sum(true_pp_original) / len(true_pp_original)
            false_perplexity_original = sum(false_pp_original) / len(false_pp_original)

            true_perplexity_llm = sum(true_pp_llm) / len(true_pp_llm)
            false_perplexity_llm = sum(false_pp_llm) / len(false_pp_llm)


            print("------")
            print(llm)
            print(dataset)
            print("Total statement (original):" + str(len(true_pp_original) + len(false_pp_original)))
            print("True statements (original):" + str(len(true_pp_original)))
            print("False statements (original):" + str(len(false_pp_original)))
            print(f"Total statement ({llm}):" + str(len(true_pp_llm) + len(false_pp_llm)))
            print(f"True statements ({llm}):" + str(len(true_pp_llm)))
            print(f"False statements ({llm}):" + str(len(false_pp_llm)))
            print("True pp original: " + str(true_perplexity_original))
            print("False pp original: " + str(false_perplexity_original))
            print(f"True pp ({llm}): " + str(true_perplexity_llm))
            print(f"False pp ({llm}): " + str(false_perplexity_llm))
            print("------")
