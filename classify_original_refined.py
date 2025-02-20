"""
Based on the original code provided by Azaria & Mitchell, 2023. (The Internal State of an LLM Knows When It's Lying)

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


import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import os


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
    val_to_ret = str_arr.replace("[array(", "").replace("dtype=float32)]", "").replace("\n","").replace(" ","").replace("],","]").replace("[","").replace("]","")
    return val_to_ret

check_uncommon = False
check_generated = False
repeat_each = args.rep

layer_num_list = [-1, -4, -8, -12, -16]
keep_probabilities = check_uncommon
check_single_first = check_uncommon or check_generated
overall_res = []

for layer_num_from_end in layer_num_list:
    dataset_names = ["capitals", "inventions", "elements", "animals", "companies", "facts"]
    results = []
    dataset_loop_length = len(dataset_names)

    for ds in range(dataset_loop_length):
        datasets = []
        if dataset_names[ds] != "facts":
            test_name = f"{llm_name}/{dataset_names[ds]}"
            test_df = pd.read_csv(f'embeddings/embeddings_with_labels_{test_name}{model_to_use}_{abs(layer_num_from_end)}_rmv_period.csv')
            dfs_to_concatenate = dataset_names[:ds] + dataset_names[ds + 1:]
            print('TEST:', test_name)
            print('TRAIN:', dfs_to_concatenate)

            for dataset_name in dfs_to_concatenate:
                datasets.append(pd.read_csv(f'embeddings/embeddings_with_labels_{dataset_name}{model_to_use}_{abs(layer_num_from_end)}_rmv_period.csv'))

            train_df = pd.concat(datasets, ignore_index=True)
            all_probs = np.zeros((len(test_df), 1))

            for i in range(repeat_each):
                train_embeddings = np.array([np.fromstring(correct_str(embedding), sep=',') for embedding in train_df['embeddings'].tolist()])
                test_embeddings = np.array([np.fromstring(correct_str(embedding), sep=',') for embedding in test_df['embeddings'].tolist()])
                train_labels = np.array(train_df['label'])
                test_labels = np.array(test_df['label'])

                model = Sequential()
                model.add(Dense(256, activation='relu', input_dim=train_embeddings.shape[1]))
                model.add(Dense(128, activation='relu'))
                model.add(Dense(64, activation='relu'))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

                model.fit(train_embeddings, train_labels, epochs=5, batch_size=32, validation_data=(test_embeddings, test_labels))
                loss, accuracy = model.evaluate(test_embeddings, test_labels)

                test_pred_prob = model.predict(test_embeddings)

                if keep_probabilities:
                    all_probs += test_pred_prob

                fpr, tpr, _ = roc_curve(test_labels, test_pred_prob)
                roc_auc = auc(fpr, tpr)
                print("AUC of the classifier on the test set:", roc_auc)

                X_val, X_test, y_val, y_test = train_test_split(test_embeddings, test_labels, test_size=0.7, random_state=42)
                y_val_pred_prob = model.predict(X_val)
                fpr_val, tpr_val, thresholds_val = roc_curve(y_val, y_val_pred_prob)

                optimal_threshold = 0.5

                y_test_pred_prob = model.predict(X_test)
                y_test_pred = (y_test_pred_prob > optimal_threshold).astype(int)

                test_accuracy = accuracy_score(y_test, y_test_pred)

                print("Optimal threshold:", optimal_threshold)
                print("Test set accuracy:", test_accuracy)

                results.append((dataset_names[ds], i, accuracy, roc_auc, optimal_threshold, test_accuracy))

            all_probs = all_probs / repeat_each
            print("----probs:----")
            print(all_probs)
            print("----end probs----")

    print(results)

    for ds in range(dataset_loop_length):
        if dataset_names[ds] != 'facts':
            relevant_results_portion = results[repeat_each * ds:repeat_each * (ds + 1)]
            acc_list = [t[2] for t in relevant_results_portion]
            auc_list = [t[3] for t in relevant_results_portion]
            opt_thresh_list = [t[4] for t in relevant_results_portion]
            acc_thr_test_list = [t[5] for t in relevant_results_portion]
            avg_acc = sum(acc_list) / len(acc_list)
            std_acc = np.std(acc_list)
            avg_auc = sum(auc_list) / len(auc_list)
            avg_thrsh = sum(opt_thresh_list) / len(opt_thresh_list)
            avg_thr_test_acc = sum(acc_thr_test_list) / len(acc_thr_test_list)
            text_res = f"dataset: {dataset_names[ds]} layer_num_from_end: {layer_num_from_end} Avg_acc: {avg_acc} Std_acc: {std_acc} Avg_AUC: {avg_auc} Avg_threshold: {avg_thrsh} Avg_thrs_acc: {avg_thr_test_acc}"
            print(text_res)
            overall_res.append(text_res)

print(overall_res)
for res in overall_res:
    print(res)


