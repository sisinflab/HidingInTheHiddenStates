import os
from datagen_perplexity.utils import get_project_root
import pandas as pd


LLMS = [
    #    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Llama-2-7b-hf",
    "facebook/opt-6.7b",
]

for llm in LLMS:
    directory = os.path.join(
        get_project_root(), "resources", llm.split("/")[-1], "triviaqa"
    )
    directory_corrected = os.path.join(
        get_project_root(), "resources", llm.split("/")[-1], "triviaqa-corrected"
    )

    dirs = [
        f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))
    ]

    for d in dirs:
        if (not os.path.exists(os.path.join(directory, d, "labeled.csv")) \
            or not os.path.exists(os.path.join(directory, d, "errors.csv")) \
            or not os.path.exists(os.path.join(directory, d, "raw.csv"))):

            raise Exception("Files missing")
        
        labeled_df = pd.read_csv(os.path.join(directory, d, "labeled.csv"), usecols=["question", "answer", "ground_truth", "eval"])
        errors_df = pd.read_csv(os.path.join(directory, d, "errors.csv"), usecols=["question", "answer", "ground_truth", "error"])
        raw_df = pd.read_csv(os.path.join(directory, d, "raw.csv"), usecols=["question", "answer", "ground_truth"])

        if (len(labeled_df) + len(errors_df)) != len(raw_df):
            raise Exception("Length mismatch")
 
        labeled_df = pd.read_csv(os.path.join(directory_corrected, d, "labeled.csv"), usecols=["question", "answer", "ground_truth", "eval"])
        raw_df = pd.read_csv(os.path.join(directory_corrected, d, "raw.csv"), usecols=["question", "answer", "ground_truth"])



        resulting_data = []
        resulting_labels = []
        print(d)

        for question, group in labeled_df.groupby("question"):
            answers = group["answer"].tolist()
            labels = group["eval"].tolist()
            for l in labels:
                try:
                    l = int(l)
                except:
                    print(group)
            filtered_answers, filtered_labels = zip(
                *[(ans, int(lbl)) for ans, lbl in zip(answers, labels) if not ans.endswith("?")]
            ) if any(not ans.endswith("?") for ans in answers) else ([], [])

            # Convert back to lists
            filtered_answers = list(filtered_answers)
            filtered_labels = list(filtered_labels)
            hit_ratio = sum(filtered_labels) / len(filtered_labels)
            if hit_ratio > 0.4 and hit_ratio < 0.6:
                resulting_data.extend(filtered_answers)
                resulting_labels.extend(filtered_labels)
#            difference = list(set(answers) - set(filtered_answers))


        df = pd.DataFrame({"statement": resulting_data, "label": resulting_labels})
        df.to_csv(os.path.join(directory, f"{d}_true_false.csv"), index=False)

        total_rows = len(df)
        positive_rows = df['label'].sum()  # Assuming positive labels are 1
        negative_rows = total_rows - positive_rows  # Assuming binary labels (1 and 0)

        print(f"{llm}, {d}")
        # Display the counts
        print(f"Total Rows: {total_rows}")
        print(f"Positive Rows: {positive_rows}")
        print(f"Negative Rows: {negative_rows}")

