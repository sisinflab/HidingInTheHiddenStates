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
import numpy as np
from transformers import AutoTokenizer, TFAutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def correct_str(str_arr):
    val_to_ret = str_arr.replace("[", "").replace("]", "").replace("\n", "").replace(" ", "").replace("array(", "").replace("dtype=float32", "")
    return val_to_ret


def str_to_embedding(str_embedding):
    # Remove brackets and spaces, and split the string by commas
    str_list = str_embedding.replace("[[", "").replace("]]", "").replace(" ", "").split(",")
    # Convert each string element to a float and create a list of floats
    embedding = [float(element) for element in str_list]
    return embedding


##Load pre-trained BERT model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# bert_model = TFAutoModel.from_pretrained('bert-base-uncased')

def get_sentence_embedding(sentence: str, tokenizer, b_model):
    inputs = tokenizer(sentence, return_tensors="tf", padding=True, truncation=True)
    outputs = b_model(inputs)
    embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()
    return [embeddings[0].tolist()]  # Convert the numpy array to a list

def create_bert_embedding(name_to_load):
    df = pd.read_csv('resources\\embeddings_with_labels_' + name_to_load + '6.7b_5fromend_rmv_period.csv')
    df['embeddings'] = df['statement'].apply(lambda x: get_sentence_embedding(x, tokenizer, bert_model))
    df.to_csv("resources\\bert_"+name_to_load+".csv", index=False) #save it
    return df

dataset_names = ["generated", "capitals", "inventions", "elements", "animals", "facts", "companies"] #"generated",
datasets = []
for dataset_name in dataset_names:
    #df = create_bert_embedding(dataset_name)
    df = pd.read_csv("D:\\datasets\\LLMTF\\bert_"+dataset_name+".csv")
    datasets.append(df)

results = []
for i in range(len(dataset_names)):
    test_df = datasets[0]#datasets[i] #i]
    #dfs_to_concatenate = datasets[:i] + datasets[i + 1:]
    dfs_to_concatenate = datasets[1:]
    train_df = pd.concat(dfs_to_concatenate, ignore_index=True)

    # Split the data into train and test sets
    #train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Extract the embeddings and labels from the train and test sets
    train_embeddings = np.array([str_to_embedding(embedding) for embedding in train_df['embeddings'].tolist()])
    test_embeddings = np.array([str_to_embedding(embedding) for embedding in test_df['embeddings'].tolist()])
    train_labels = np.array(train_df['label'])
    test_labels = np.array(test_df['label'])

    print(train_embeddings.shape[1])

    # Define the neural network model
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=train_embeddings.shape[1])) #change input_dim to match the number of elements in train_embeddings...
    model.add(Dense(128, activation='relu'))
    #model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # Compile the model

    # Train the model
    model.fit(train_embeddings, train_labels, epochs=5, batch_size=32, validation_data=(test_embeddings, test_labels))
    loss, accuracy = model.evaluate(test_embeddings, test_labels)

    test_pred_prob = model.predict(test_embeddings)

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(test_labels, test_pred_prob)  # Assuming binary classification
    roc_auc = auc(fpr, tpr)
    print("AUC of the classifier on the test set:", roc_auc)

    results.append((i,accuracy, roc_auc))
    #results.append((dataset_names[i],accuracy))

print(results)
# Extract the second item from each tuple and put it in a list
acc_list = [t[1] for t in results]
auc_list = [t[2] for t in results]
# Calculate the average of the numbers in the list
avg_acc = sum(acc_list) / len(acc_list)
avg_auc = sum(auc_list) / len(auc_list)
print("Avg_acc:" + str(avg_acc) + " Avg_AUC:" + str(avg_auc))





