import pandas as pd
import os
from datagen_perplexity.utils import get_project_root
from datagen_perplexity.utils import get_llm, perplexity_based_sampling
from datagen_perplexity.utils import enable_determinism
import tqdm
import numpy as np


def main(llm_name, model, tokenizer):
    # Read the CSV file
    df = pd.read_csv(
        os.path.join(get_project_root(), "resources", "worldcities_filtered.csv")
    )

    is_city_in_str = " is a city in "
    is_capital_of = " is the capital of "
    # is_city_name = " is a name of a city."
    has_population = " has a population of approximately "
    # is_country_name = " is a name of a country."
    # List to store the final dataset
    dataset = []

    df = df.sample(750)

    # Loop through each row in the dataframe
    for i, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        # Extract the country and capital city from the row
        country = row["country"]
        city = row["city"]
        population = row["population"]
        is_capital = row["is_capital"]

        # city is in {country}

        template = city + is_city_in_str + "{0}" + "."
        result = perplexity_based_sampling(
            required_param="country",
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

        # TODO: doesn't scale
        # # city has population of {#}
        # if not np.isnan(population):
        #     sentence = city + has_population + str(int(population)) + "."
        #     dataset.append((sentence, 1))
        #
        #     template = city + has_population + "{0}" + "."
        #     sentence, prob, _, _ = perplexity_based_sampling(required_param="population", current_val=population,
        #                                                      template=template, df=df, model=model, tokenizer=tokenizer)
        #     dataset.append((sentence, 0))
        #     print(sentence)
        #     print(prob)

        # TODO: there is an error in Azaria
        # capital_label = 0 if is_capital == "" else 1
        #
        # if capital_label == "primary" or capital_label == "":
        #     sentence = city + is_capital_of + country + "."
        #     if capital_label == 1:
        #     dataset.append((sentence, capital_label))

        # # add four more sentences two true and two false
        # if country != city: #make sure the city and country have different names (e.g. San Marino)
        #     sentence = city + is_city_name
        #     dataset.append((sentence, 1))
        #     sentence = city + is_country_name
        #     dataset.append((sentence, 0))
        #
        #     sentence = country + is_city_name
        #     dataset.append((sentence, 0))
        #     sentence = country + is_country_name
        #     dataset.append((sentence, 1))

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
            "capitals_true_false.csv",
        ),
        index=False,
    )  # , header=False)
