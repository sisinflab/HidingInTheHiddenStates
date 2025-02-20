from datasets import load_dataset
import torch
import re
import tldextract
from datagen_perplexity.utils import get_llm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
from datagen_perplexity.utils import get_project_root
from datagen_perplexity.utils import enable_determinism
from collections import Counter
import pandas as pd
import numpy as np
import os

def process_url(url):
    extracted = tldextract.extract(url)
    second_level = extracted.domain
    third_level = extracted.subdomain if extracted.subdomain else ""  # Check if a third-level domain exists
    
    return third_level + second_level

def annotate_answer(ans, gt):

    gt_aliases = gt["aliases"] + gt["normalized_aliases"]

    return int(any(re.search(r'\b' + re.escape(alias) + r'\b', ans) for alias in gt_aliases))



enable_determinism(deterministic_algorithms=False)

LLMS = [
    "facebook/opt-6.7b",
    "meta-llama/Llama-2-7b-hf",
]

# Load the TriviaQA dataset and select the required subset
dataset = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext", split="validation")

# Count occurrences of each unique question source
question_source_counts = Counter(row['question_source'] for row in dataset)

print(question_source_counts)

for source, count in question_source_counts.items():
    print(f"{source}: {count}")

    # Filter the rows where the specific column has the value "http://www.triviacountry.com/"
    #filtered_dataset = dataset.filter(lambda row: row['question_source'] == "www.sfquiz.org.uk")
    dataset_partition = source

    filtered_dataset = dataset.filter(lambda row: row['question_source'] == dataset_partition)

    # Display some details about the filtered dataset
    print(f"Filtered dataset contains {len(filtered_dataset)} rows.")


    for llm_name in LLMS:


        enable_determinism(deterministic_algorithms=False)

        dest_dir = os.path.join(get_project_root(), "resources", llm_name.split("/")[-1], "triviaqa", process_url(dataset_partition))

        if os.path.exists(dest_dir):
            print(f"QA dataset already exists: f{dest_dir}")
            continue
        generated_dataset = []
        annotated_dataset = []

        #"facebook/opt-6.7b"
        llm, tokenizer = get_llm(llm_name)
        #llm, tokenizer = get_llm("meta-llama/Llama-2-7b-hf")

        # Prepare the LLM pipeline
        text_generator = pipeline("text-generation", model=llm, tokenizer=tokenizer)

        terminators = [
            tokenizer.eos_token_id,
        ]


        examples = """
        Question: Where in England was Dame Judi Dench born?
        Answer: The English actress Dame Judi Dench was born in York, England.

        Question: From which country did Angola achieve independence in 1975?
        Answer: Angola achieved independence from Portugal in 1975.

        Question: Which city does David Soul come from?
        Answer: David Soul hails from Chicago, Illinois.

        Question: Who won Super Bowl XX?
        Answer: The Chicago Bears won Super Bowl XX.

        Question: Which was the first European country to abolish capital punishment?
        Answer: Norway was the first European country to abolish capital punishment.

        Question: In which country did the widespread use of ISDN begin in 1988?
        Answer: The widespread use of ISDN began in Japan in 1988.

        Question: What is Bruce Willis' real first name?
        Answer: Bruce Willis' real first name is Walter.

        Question: Which William wrote the novel Lord of the Flies?
        Answer: The William who wrote Lord of the Flies was William Golding.

        Question: How is Joan Molinsky better known?
        Answer: Joan Molinsky is better known as Joan Rivers.

        Question: In which branch of the arts is Patricia Neary famous?
        Answer: Patricia Neary is famous in the field of ballet.
        """

        print(filtered_dataset)

        for q in tqdm(filtered_dataset):

            # Get the first question
            curr_question = q['question'] 
            ground_truth = q['answer'] 

            # Generate a response for the first question
            response = text_generator(
                examples + f"Question: {curr_question} \nAnswer:", 
                max_new_tokens=128,
                eos_token_id=terminators,
                stop_strings=["\nQuestion:"],
                tokenizer=tokenizer,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=1.,
                num_return_sequences=10,
            )

            answers = []
            for r in response:
                ans = r['generated_text'].split(f"Question: {curr_question} \nAnswer:")[1].split("Question:")[0].split("\n")[0].strip()
        #        if len(ans.split(" ")) < len(curr_question.split(" ")) / 2:
                if len(ans.split(" ")) < 5: # we filter out short answers
                    continue
                answers.append(ans)

            answers = np.unique(answers)
    
            if len(answers) < 5: # less than 5 "long" answers, we skip
                continue

            labels = []
            for ans in answers:
                label = annotate_answer(ans, ground_truth)
                labels.append(label)
                generated_dataset.append((curr_question, ans, ground_truth))

            hit_rate = sum(labels) / len(labels)
            if hit_rate > 0.4 and hit_rate < 0.6:
                for i, ans in enumerate(answers):
                    annotated_dataset.append((ans, labels[i]))


        
        # end of generation

        os.makedirs(
            dest_dir,
            exist_ok=True,
        )

        raw_df = pd.DataFrame(generated_dataset, columns=["question", "answer", "ground_truth"])
        df = pd.DataFrame(annotated_dataset, columns=["statement", "label"])

        raw_df.to_csv(
            os.path.join(
                dest_dir,
                "raw.csv",
            ), index=False,
        )

        df.to_csv(
            os.path.join(
                dest_dir,
                "data.csv",
            ), index=False,
        )

        torch.cuda.empty_cache()
