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


import torch
import re
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def get_period_spaces_mask(text):
    period_indices = [i for i, char in enumerate(text) if char == '.' and (len(text) == i+1 or text[i+1] != '.')] #... should be counted as a single period
    mask = [len(text) == i+1 or text[i+1] == ' ' or text[i+1] == '\n' for i in period_indices]
    return mask #if the last token is a period, and there is no next token (not even sentence end), there will be one too many masks


import numpy
from transformers import AutoTokenizer, OPTForCausalLM# AutoModelForCausalLM
#from tensorflow.keras.models import load_model


model_to_use = "6.7b"
datasets_to_use = ["companies"]

model = OPTForCausalLM.from_pretrained("facebook/opt-" + model_to_use)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-" + model_to_use)

#start_pos is used to ignore the words "It is true that" and "It is false that"
def sent_prob(sentence, start_pos=5):
    global i, inputs, text, sen_prod
    pmpt_as_token_ids = tokenizer(sentence).input_ids
    pmpt_as_tokens = [tokenizer.convert_ids_to_tokens(token_id) for token_id in pmpt_as_token_ids]
    probs = []
    for i in range(start_pos, len(pmpt_as_tokens) - 1):
        tokens_to_use = pmpt_as_tokens[1:i + 1]
        # print(tokens_to_use)
        # Convert tokens back to a sentence
        pmpt_to_use = tokenizer.convert_tokens_to_string(tokens_to_use)
        # Tokenize the prompt
        inputs = tokenizer(pmpt_to_use, return_tensors="pt")

        # Generate text
        outputs = model.generate(inputs.input_ids, return_dict_in_generate=True, max_new_tokens=1, min_new_tokens=1,
                                 output_scores=True, no_repeat_ngram_size=3, output_hidden_states=True)
        generate_ids = outputs.sequences  # outputs[0]
        text = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        # print(text)
        token_to_eval = pmpt_as_tokens[i + 1]
        probabilities = torch.softmax(outputs.scores[-1][0], dim=-1)
        # prob = outputs.scores[-1][0][tokenizer.get_vocab()[token_to_eval]].item()
        prob = probabilities[tokenizer.get_vocab()[token_to_eval]].item()
        # print(token_to_eval + ": " + str(prob))
        probs.append(prob)
    sen_prod = np.prod(np.array((probs)))
    return sen_prod

correct_count = 0
total_count = 0
for dataset in datasets_to_use:
    df = pd.read_csv("resources\\" + dataset + "_true_false.csv")
    df['prob_true'] = pd.Series(dtype=float)
    df['prob_false'] = pd.Series(dtype=float)
    for r_idx, row in df.iterrows():
        prompt = row['statement']
        sentence = "It is true that " + prompt
        it_is_true_prob = sent_prob(sentence)
        df.at[r_idx, 'prob_true'] = it_is_true_prob
        sentence = "It is false that " + prompt
        it_is_false_prob = sent_prob(sentence)
        df.at[r_idx, 'prob_false'] = it_is_false_prob
        label = row['label']
        correct = it_is_true_prob > it_is_false_prob and label == 1 or it_is_true_prob <= it_is_false_prob and label == 0
        if correct:
            correct_count += 1
        total_count += 1
        print(str(total_count) + " (" +str(correct_count) + "). " + prompt + "---True Probablity: " + str(it_is_true_prob) + ". False: " + str(it_is_false_prob) + ". Correct: " + str(correct))
    df.to_csv("D:\\datasets\\LLMTF\\" + "it_is_true_false_model_probs_" + dataset + model_to_use + ".csv",
              index=False)
print("correct count: " + str(correct_count) + " total:" + str(total_count) + " frac:" + str(correct_count/total_count))
