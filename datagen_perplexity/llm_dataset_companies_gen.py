import pandas as pd
from datagen_perplexity.utils import get_project_root
from datagen_perplexity.utils import get_llm, perplexity_based_sampling
from datagen_perplexity.utils import enable_determinism
import os
import tqdm


def main(llm_name, model, tokenizer):
    # Read the CSV file

    df = pd.read_csv(os.path.join(get_project_root(), "resources", "companies.csv"))

    dataset = []

    for i, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        # Extract the country and capital city from the row
        name = row["Company Name"]
        country = row["Country"]
        industry = row["Industry"].lower()
        info = row["Additional Info"]

        # Country
        has_headquarters_in = " has headquarters in "
        template = name + has_headquarters_in + "{0}" + "."
        result = perplexity_based_sampling(
            required_param="Country",
            current_val=country,
            template=template,
            df=df,
            model=model,
            tokenizer=tokenizer,
        )
        if result is not False:
            dataset.append((template.format(country), 1))
            sentence, prob, _, _ = result
            dataset.append((sentence, 0))
            print(sentence)
            print(prob)

        # Industry
        operates_in_industry = " operates in the industry of "
        template = name + operates_in_industry + "{0}" + "."
        result = perplexity_based_sampling(
            required_param="Industry",
            current_val=industry,
            template=template,
            df=df,
            model=model,
            tokenizer=tokenizer,
        )
        if result is not False:
            dataset.append((template.format(industry), 1))
            sentence, prob, _, _ = result
            dataset.append((sentence, 0))
            print(sentence)
            print(prob)

        # Additional info

        template = name + " " + "{0}" + "."
        result = perplexity_based_sampling(
            required_param="Additional Info",
            current_val=info,
            template=template,
            df=df,
            model=model,
            tokenizer=tokenizer,
        )
        if result is not False:
            dataset.append((template.format(info), 1))
            sentence, prob, _, _ = result
            dataset.append((sentence, 0))
            print(sentence)
            print(prob)

    # Shuffle the dataset
    import random

    random.shuffle(dataset)

    # Print the first 10 examples in the dataset
    print(dataset[:30])
    dataset_df = pd.DataFrame(dataset, columns=["statement", "label"])
    os.makedirs(
        os.path.join(get_project_root(), "resources", llm_name.split("/")[-1]),
        exist_ok=True,
    )
    dataset_df.to_csv(
        os.path.join(
            get_project_root(),
            "resources",
            llm_name.split("/")[-1],
            "companies_true_false.csv",
        ),
        index=False,
    )  # , header=False)
