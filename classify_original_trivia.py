"""
Original code provided by Azaria & Mitchell, 2023.

@inproceedings{azaria-mitchell-2023-internal,
    title = "The Internal State of an {LLM} Knows When It`s Lying",
    author = "Azaria, Amos  and
      Mitchell, Tom",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.68/",
    doi = "10.18653/v1/2023.findings-emnlp.68",
    pages = "967--976",
    abstract = "While Large Language Models (LLMs) have shown exceptional performance in various tasks, one of their most prominent drawbacks is generating inaccurate or false information with a confident tone. In this paper, we provide evidence that the LLM`s internal state can be used to reveal the truthfulness of statements. This includes both statements provided to the LLM, and statements that the LLM itself generates. Our approach is to train a classifier that outputs the probability that a statement is truthful, based on the hidden layer activations of the LLM as it reads or generates the statement. Experiments demonstrate that given a set of test sentences, of which half are true and half false, our trained classifier achieves an average of 71{\%} to 83{\%} accuracy labeling which sentences are true versus false, depending on the LLM base model. Furthermore, we explore the relationship between our classifier`s performance and approaches based on the probability assigned to the sentence by the LLM. We show that while LLM-assigned sentence probability is related to sentence truthfulness, this probability is also dependent on sentence length and the frequencies of words in the sentence, resulting in our trained classifier providing a more reliable approach to detecting truthfulness, highlighting its potential to enhance the reliability of LLM-generated content and its practical applicability in real-world scenarios."
}
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
import tensorflow as tf
import argparse

# Define argument parser
parser = argparse.ArgumentParser(description="Parameterize LLM choice.")
parser.add_argument("--llm", type=str, choices=["llama", "opt"], required=True, help="Specify the LLM to use: 'llama' or 'opt'.")
parser.add_argument("--rep", type=int, default=20)
args = parser.parse_args()

# Adjust model_to_use and test_name based on llm argument
if args.llm == "llama":
    model_to_use = "LLAMA7"
    llm_name = "LLama-2-7b-hf"
elif args.llm == "opt":
    model_to_use = "6.7b"
    llm_name = "opt-6.7b"
else:
    model_to_use = "6.7b"
    llm_name = "opt-6.7b"


def correct_str(str_arr):
    val_to_ret = str_arr.replace("[array(", "").replace("dtype=float32)]", "").replace("\n", "").replace(" ", "").replace("],", "]").replace("[", "").replace("]", "")
    return val_to_ret

layer_num_list = [-1, -4, -8, -12, -16]
repeat_each = args.rep

train_dataset_names = ["capitals", "inventions", "elements", "animals", "companies", "facts"]
test_dataset_names = [
    f"{llm_name}/triviaqa/billturnbullquiz4free",
    f"{llm_name}/triviaqa/derbyshirepubquizleaguewordpress",
    f"{llm_name}/triviaqa/quiz4free",
    f"{llm_name}/triviaqa/quizguywordpress",
    f"{llm_name}/triviaqa/triviabug",
    f"{llm_name}/triviaqa/wwwbusinessballs",
    f"{llm_name}/triviaqa/wwwjetpunk",
    f"{llm_name}/triviaqa/wwwodquiz",
    f"{llm_name}/triviaqa/wwwquiz-zone",
    f"{llm_name}/triviaqa/wwwquizballs",
    f"{llm_name}/triviaqa/wwwquizwise",
    f"{llm_name}/triviaqa/wwwsfquiz",
    f"{llm_name}/triviaqa/wwwtriviacountry",
    f"{llm_name}/triviaqa/wwwwrexhamquizleague"
]

start_time = time.time()
overall_res = []

for layer_num_from_end in layer_num_list:
    train_dfs = [pd.read_csv(f'embeddings/embeddings_with_labels_{name}{model_to_use}_{abs(layer_num_from_end)}_rmv_period.csv') for name in train_dataset_names]
    train_df = pd.concat(train_dfs, ignore_index=True)

    train_embeddings = np.array([np.fromstring(correct_str(embedding), sep=',') for embedding in train_df['embeddings'].tolist()])
    train_labels = np.array(train_df['label'])

    model = Sequential([
        Dense(256, activation='relu', input_dim=train_embeddings.shape[1]),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_embeddings, train_labels, epochs=5, batch_size=32, validation_split=0.2)

    for test_dataset in test_dataset_names:
        test_df = pd.read_csv(f'embeddings/embeddings_with_labels_{test_dataset}{model_to_use}_{abs(layer_num_from_end)}_rmv_period.csv')
        test_embeddings = np.array([np.fromstring(correct_str(embedding), sep=',') for embedding in test_df['embeddings'].tolist()])
        test_labels = np.array(test_df['label'])

        all_probs = np.zeros((len(test_df), 1))
        results = []

        for i in range(repeat_each):
            loss, accuracy = model.evaluate(test_embeddings, test_labels, verbose=0)
            test_pred_prob = model.predict(test_embeddings, verbose=0)
            fpr, tpr, _ = roc_curve(test_labels, test_pred_prob)
            roc_auc = auc(fpr, tpr)

            X_val, X_test, y_val, y_test = train_test_split(test_embeddings, test_labels, test_size=0.7, random_state=42)
            y_val_pred_prob = model.predict(X_val, verbose=0)
            fpr_val, tpr_val, thresholds_val = roc_curve(y_val, y_val_pred_prob)

            optimal_threshold = thresholds_val[np.argmax([accuracy_score(y_val, y_val_pred_prob > thr) for thr in thresholds_val])]
            
            y_test_pred_prob = model.predict(X_test, verbose=0)
            y_test_pred = (y_test_pred_prob > optimal_threshold).astype(int)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            results.append((test_dataset, i, accuracy, roc_auc, optimal_threshold, test_accuracy))
            all_probs += test_pred_prob

        all_probs /= repeat_each
        acc_list = [t[2] for t in results]
        auc_list = [t[3] for t in results]
        opt_thresh_list = [t[4] for t in results]
        acc_thr_test_list = [t[5] for t in results]

        avg_acc = np.mean(acc_list)
        std_acc = np.std(acc_list)
        avg_auc = np.mean(auc_list)
        avg_thrsh = np.mean(opt_thresh_list)
        avg_thr_test_acc = np.mean(acc_thr_test_list)

        text_res = f"dataset: {test_dataset} layer_num_from_end: {layer_num_from_end} Avg_acc: {avg_acc} Std_acc: {std_acc} Avg_AUC: {avg_auc} Avg_threshold: {avg_thrsh} Avg_thrs_acc: {avg_thr_test_acc}"
        print(text_res)
        overall_res.append(text_res)

end_time = time.time()
print(f"Training took {end_time - start_time} seconds.")
