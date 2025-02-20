from datagen_perplexity.utils import enable_determinism
import tiktoken
from datagen_perplexity.utils import get_project_root
import pandas as pd
import json
from tqdm import tqdm
import os
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


endpoint = os.getenv("ENDPOINT_URL", "")
deployment = os.getenv("DEPLOYMENT_NAME", "")
tiktoken_encoder = tiktoken.encoding_for_model(deployment)

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key="",
    api_version="2024-08-01-preview",
)


SYSTEM_PROMPT = """You are a judge and your role is to judge whether the provided answer is correct for the given question, based on the provided ground truth. Answer with a 1 if the answer is correct and 0 if the answer is incorrect.
Here are a few examples: """


def construct_user_prompt(question, answer, ground_truth):
    prompt = f"""
Question: {question}
Answer: {answer}
Ground truth: {ground_truth}
Evaluation: 
"""

    return prompt


examples = [
    {
        "question": "Who was the next British Prime Minister after Arthur Balfour?",
        "answer": "Arthur Balfour was followed by David Lloyd George.",
        "ground_truth": {
            "aliases": [
                "Sir Henry Campbell-Bannerman",
                "Campbell-Bannerman",
                "Campbell Bannerman",
                "Sir Henry Campbell Bannerman",
                "Henry Campbell Bannerman",
                "Henry Campbell-Bannerman",
            ],
            "normalized_aliases": [
                "henry campbell bannerman",
                "sir henry campbell bannerman",
                "campbell bannerman",
            ],
            "matched_wiki_entity_name": "",
            "normalized_matched_wiki_entity_name": "",
            "normalized_value": "campbell bannerman",
            "type": "WikipediaEntity",
            "value": "Campbell-Bannerman",
        },
        "evaluation": 0,
    },
    {
        "question": "Who had a 70s No 1 hit with Kiss You All Over?",
        "answer": "The band Exile had a 70s No 1 hit with Kiss You All Over.",
        "ground_truth": {
            "aliases": [
                "Internal exile",
                "Exiles",
                "Transported for life",
                "Exile (politics and government)",
                "Voluntary exile",
                "Sent into exile",
                "Exile and Banishment",
                "Self-exile",
                "Forced exile",
                "Exile",
                "Exile in Greek tragedy",
                "Banish",
                "Banishment",
            ],
            "normalized_aliases": [
                "exiles",
                "voluntary exile",
                "forced exile",
                "banish",
                "self exile",
                "exile politics and government",
                "exile in greek tragedy",
                "sent into exile",
                "banishment",
                "transported for life",
                "exile",
                "internal exile",
                "exile and banishment",
            ],
            "matched_wiki_entity_name": "",
            "normalized_matched_wiki_entity_name": "",
            "normalized_value": "exile",
            "type": "WikipediaEntity",
            "value": "Exile",
        },
        "evaluation": 1,
    },
    {
        "question": " Which common mineral is used to make casts, moulds, blackboard chalk and plaster of Paris?",
        "answer": "The common mineral used to make casts, moulds, blackboard chalk and plaster of Paris is calcium carbonate.",
        "ground_truth": {
            "aliases": [
                "CaSO4·2H2O",
                "Gypsum",
                "Calcium sulfate dihydrate",
                "CaSO4*2H2O",
                "Gipsum",
            ],
            "normalized_aliases": [
                "calcium sulfate dihydrate",
                "caso4 2h2o",
                "gipsum",
                "caso4·2h2o",
                "gypsum",
            ],
            "matched_wiki_entity_name": "",
            "normalized_matched_wiki_entity_name": "",
            "normalized_value": "gypsum",
            "type": "WikipediaEntity",
            "value": "Gypsum",
        },
        "evaluation": 0,
    },
]


def structure_user_prompt(question, answer, gt):
    return f"""
Question: {question}  
Answer: {answer}
Ground truth: {json.dumps(gt)}
Evaluation: 
    """


def query_gpt4o_mini(prefix, question, answer, gt):

    messages = prefix + [{"role": "user", "content": structure_user_prompt(question, answer, gt)}]

    num_tokens_from_string = compute_tokens(json.dumps(messages))
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
        )
        return response.choices[0].message.content, num_tokens_from_string, None
    except Exception as e:
        print(f"Error querying GPT: {e}")
        return None, num_tokens_from_string, e


enable_determinism(deterministic_algorithms=False)
LLMS = [
    #    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Llama-2-7b-hf",
    "facebook/opt-6.7b",
]

def compute_tokens(string: str) -> int:
    num_tokens = len(tiktoken_encoder.encode(string))
    return num_tokens


total_tokens = 0
message_prefix = [{"role": "system", "content": SYSTEM_PROMPT}]

for e in examples:
    message_prefix.append(
        {
            "role": "user",
            "content": structure_user_prompt(
                e["question"], e["answer"], e["ground_truth"]
            ),
        }
    )
    message_prefix.append({"role": "assistant", "content": str(e["evaluation"])})

for llm in LLMS:

    directory = os.path.join(
        get_project_root(), "resources", llm.split("/")[-1], "triviaqa"
    )

    dirs = [
        f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))
    ]

    for d in dirs:
        print(f"Computing {os.path.join(directory, d, "labeled.csv")}")
        if os.path.exists(os.path.join(directory, d, "labeled.csv")):
            pass
           # continue
        chatgpt_results = []
        error_results = []
        raw_file = os.path.join(directory, d, "raw.csv")
        df = pd.read_csv(raw_file)

        for index, row in tqdm(df.iterrows()):
            res, toks, err = query_gpt4o_mini(message_prefix, row["question"], row["answer"], row["ground_truth"])
            total_tokens += toks
            if res is None:
                error_results.append((
                    row["question"],
                    row["answer"],
                    row["ground_truth"],
                    err
                ))
            else:
                chatgpt_results.append((
                    row["question"],
                    row["answer"],
                    row["ground_truth"],
                    res,
                ))
            print(total_tokens)

            if index % 10 == 0:
                labeled_df = pd.DataFrame(chatgpt_results, columns=["question", "answer", "ground_truth", "eval"])
                errors_df = pd.DataFrame(error_results, columns=["question", "answer", "ground_truth", "error"])
                labeled_df.to_csv(
                    os.path.join(directory, d, "labeled.csv"), index=False,
                )
                errors_df.to_csv(
                    os.path.join(directory, d, "errors.csv"), index=False,
                )
        labeled_df = pd.DataFrame(chatgpt_results, columns=["question", "answer", "ground_truth", "eval"])
        errors_df = pd.DataFrame(error_results, columns=["question", "answer", "ground_truth", "error"])
        labeled_df.to_csv(
            os.path.join(directory, d, "labeled.csv"), index=False,
        )
        errors_df.to_csv(
            os.path.join(directory, d, "errors.csv"), index=False,
        )
