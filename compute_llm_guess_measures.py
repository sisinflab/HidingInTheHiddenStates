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

import pandas as pd
from sklearn.metrics import auc, roc_curve

num_of_few_shot = 5
topic = "generated"
# Read the CSV file
file_path = 'D:\\datasets\\LLMTF\\llm_guess_'+topic+'6.7b_'+str(num_of_few_shot) + ".csv"
data = pd.read_csv(file_path)

# Remove the first num_of_few_shot rows without predictions
data = data.iloc[num_of_few_shot:]

# Extract the true labels and logits
true_labels = data['label']
logits = data['true-false']

# Compute ROC curve and ROC area using logits
fpr, tpr, _ = roc_curve(true_labels, logits)
roc_auc = auc(fpr, tpr)

print("AUC of the classifier using logits:", roc_auc)
