# Are the Hidden States Hiding Something?

### Testing the Limits of Factuality-Encoding Capabilities in LLMs

This repository contains the codebase used in the paper _"Are the Hidden States Hiding Something? Testing the Limits of Factuality-Encoding Capabilities in LLMs"_.

## Overview

This work extends the study by Azaria & Mitchell, [_The Internal State of an LLM Knows When It's Lying_](https://aclanthology.org/2023.findings-emnlp.68/). We reproduce their experiments and further explore the capabilities of SAPLMA on a newly generated dataset. Specifically, we:

- Refine their True-False dataset using a perplexity-based sampling strategy.
- Generate a novel fact collection derived from a well-known Question Answering dataset.

We build upon the code provided by Azaria & Mitchell, making only minor modifications to their codebase.

---

## Reproducing Experimental Results

### 1. Create and Activate a Virtual Environment

```sh
python -m venv HiddenStates
source HiddenStates/bin/activate
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

### 3. Set Up Hugging Face Token

Log in to Hugging Face using the CLI and set up your token:

```sh
huggingface-cli login
```

You can obtain a token from your [Hugging Face account settings](https://huggingface.co/settings/tokens).

### 4. Generate the Datasets

1. Instructions for downloading the original True-False dataset are available in `resources/TrueFalse Dataset.md`.
2. Generate the refined True-False dataset:

```sh
python llm_dataset_gen.py
```

1. Generate the TriviaQA-based fact collection:

```sh
python llm_datagen_qa.py
python chatgpt_check_qa.py
python parse_chagpt_datasets.py
```

We provide the dataset used in our experiments in the directories:
```
resource/
├── Llama-2-7b-hf/
│   ├── [Refined True-False for Llama-2]
│   ├── triviaqa/
│   │   ├── [Collections of facts, extracted from TriviaQA via Llama-2]
│
├── opt-6.7b/
│   ├── [Refined True-False for OPT-6.7b]
│   ├── triviaqa/
│   │   ├── [Collections of facts, extracted from TriviaQA via OPT-6.7b]
```
### 5. Run the Experiments

Make sure to adjust paths if needed before running the following scripts.

---

## Experiments

### Generating Hidden States

To correctly launch the experiments, generate the hidden states:

```sh
python gen_embedding.py
```

Specify the dataset and model within the script.

### Running SAPLMA Classifications

#### Train and test SAPLMA on the same dataset:

```sh
python classify_sentences.py --llm {opt|llama}
```

#### Train on the original True-False dataset and test on the refined version:

```sh
python classify_original_refined.py --llm {opt|llama}
```

#### Train on the original True-False dataset and test on TriviaQA-extracted facts:

```sh
python classify_original_trivia.py --llm {opt|llama}
```


---

## Baselines

### Evaluating Sentences with BERT

```sh
python classify_bert_sentences.py
```

Note: The lines for generating BERT hidden states are commented out in the script. Uncomment them before first usage.

### Evaluating the _Is-it-True_ Baseline

```sh
python llmGProbItIssTrueFalse.py
```

### Evaluating the 3-shot and 5-shot Baselines

```sh
python llmEvaluate.py
python compute_llm_guess_measure.py
```

Modify the number of shots within the script as needed.

---

## Information Extraction
It is possible to compute the average perplexity of each dataset via

```
python compute_perplexity.py
```

---

This repository provides all necessary resources to replicate and extend the experiments. If you encounter any issues, feel free to reach out!

## Acknowledgments 
We thank authors of the work we reproduce.
```
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
}
```
## Contact us
Our contact information will be added upon acceptance.