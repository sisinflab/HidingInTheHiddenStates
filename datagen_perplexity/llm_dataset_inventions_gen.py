import pandas as pd
import os
from datagen_perplexity.utils import get_project_root
from datagen_perplexity.utils import perplexity_based_sampling
import tqdm


def get_name_country_invention(row):
    name = row["Name"]
    # years = row["Years"]
    country = row["Country"]
    invention: str = row["Invention"]
    if invention.endswith("."):
        invention = invention[:-1]

    # birth_year, death_year = get_birth_and_death_years(years)
    # return name.strip(), birth_year.strip(), death_year.strip(), country.strip(), invention.strip()
    return name.strip(), country.strip(), invention.strip()


def add_sentence(dataset_to_add_to, sent, label):
    dataset_to_add_to.append((sent.replace("..", "."), label))


def main(llm_name, model, tokenizer):
    # Read the CSV file

    df = pd.read_csv(os.path.join(get_project_root(), "resources", "inventions.csv"))

    invented_text = " invented the "
    lived_text = " lived in "

    # List to store the final dataset
    dataset = []

    # Loop through each row in the dataframe
    for i, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        # name, birth_year, death_year, country, invention = get_name_birth_death_country_invention(row)
        name, country, invention = get_name_country_invention(row)

        template = name + invented_text + "{0}" + "."
        result = perplexity_based_sampling(
            required_param="Invention",
            current_val=invention,
            template=template,
            df=df,
            model=model,
            tokenizer=tokenizer,
        )
        if result is not False:
            add_sentence(dataset, template.format(invention), 1)
            sentence, prob, _, _ = result
            add_sentence(dataset, sentence, 0)
            print(sentence)
            print(prob)

        template = name + lived_text + "{0}" + "."
        result = perplexity_based_sampling(
            required_param="Country",
            current_val=country,
            template=template,
            df=df,
            model=model,
            tokenizer=tokenizer,
        )
        if result is not False:
            add_sentence(dataset, template.format(country), 1)
            sentence, prob, _, _ = result
            add_sentence(dataset, sentence, 0)

            print(sentence)
            print(prob)

    # Shuffle the dataset
    import random

    random.shuffle(dataset)

    # Print the first 10 examples in the dataset
    print(dataset[:10])
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
            "inventions_true_false.csv",
        ),
        index=False,
    )  # , header=False)
