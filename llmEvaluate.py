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


from transformers import AutoTokenizer, OPTForCausalLM# AutoModelForCausalLM
model_to_use = "6.7b" #"6.7b" "2.7b" "1.3b" "350m"

import pandas as pd
import numpy as np

num_of_examples = 3

model = OPTForCausalLM.from_pretrained("facebook/opt-"+model_to_use)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-"+model_to_use)

list_of_datasets = ["generated"] #, "capitals", "inventions", "elements", "animals", "facts", "companies"]
for dataset_to_use in list_of_datasets:
#dataset_to_use = "generated" #"math" #"capitals" "animals" "elements" "grammar" #"inventions" #"cities"#"facts" #"colors"
    df = pd.read_csv("resources\\" + dataset_to_use + "_true_false.csv")#.head(1000)
    df['next_id'] = pd.Series(dtype=float)


    def statement_and_true_false(df, loc):
        sentence = df.at[loc, 'statement']
        sentence = sentence.rstrip(". ")
        sentence += ": "
        truth = df.at[loc, 'label']
        sentence += "true. " if truth == 1 else "false. "
        return sentence

    for i in range(num_of_examples, len(df)):
        prompt = ""
        for j in range(num_of_examples-1, 0, -1):
            prompt += statement_and_true_false(df, i-j)
        prompt = df.at[i, 'statement'] #"London is"#"London, which was once located in the UK, is now located in Pakistan." #"The following is a Q&A between two math experts. Q: How much is 1+1? A: 2. Q: How much is 2+2? A:"
        prompt = prompt.rstrip(". ")
        prompt += ": "
        inputs = tokenizer(prompt, return_tensors="pt")

        outputs = model.generate(inputs.input_ids, output_hidden_states=True, return_dict_in_generate=True, max_new_tokens=1, min_new_tokens=1, output_scores=True)#, max_new_tokens=5, min_new_tokens=1) # return_logits=True, max_length=5, min_length=5, do_sample=True, temperature=0.5, no_repeat_ngram_size=3, top_p=0.92, top_k=10)return_logits=True
        generate_ids = outputs[0]
        next_id = np.array(generate_ids)[0][-1]
        df.at[i, 'next_id'] = next_id
        df.at[i, 'true'] = outputs.scores[-1][0][tokenizer.get_vocab()['true']].item()
        df.at[i, 'false'] = outputs.scores[-1][0][tokenizer.get_vocab()['false']].item()
        print("processing: " + str(i) + ", next_token:" + str(next_id))

    df.to_csv("resources\\" + "llm_guess_" + dataset_to_use + model_to_use + "_" + str(num_of_examples) + ".csv", index=False)

#results
#capitals (by subtracting true from false and subtracting the average): 0.54158
#facts: 0.51639
#animals: 0.56517419
#elements: 0.568500539
#inventions: 0.4799954
#generated: 0.46124031
#copanies: 0.5538847
#average: 0.52419752

