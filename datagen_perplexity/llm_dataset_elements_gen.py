import pandas as pd
from datagen_perplexity.utils import get_project_root
from datagen_perplexity.utils import get_llm, perplexity_based_sampling
from datagen_perplexity.utils import enable_determinism
import os
import tqdm

# Read the CSV file


def main(llm_name, model, tokenizer):
    # Read the CSV file

    df = pd.read_csv(os.path.join(get_project_root(), "resources", "elements.csv"))

    dataset = []
    # AtomicNumber	Symbol	Name	StandardState	MeltingPoint	GroupBlock	YearDiscovered	info

    # Loop through each row in the dataframe
    def get_year_info(year_discovered):
        if year_discovered == "Ancient":
            return "was discovered over a thousand years ago"
        return "was discovered in " + str(year_discovered)

    for i, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        # Extract the country and capital city from the row
        name = row["Name"]
        atomic_number = row["AtomicNumber"]
        symbol = row["Symbol"]
        standard_state = row["StandardState"]
        # melting_point = row["MeltingPoint"]
        group_block = row["GroupBlock"]
        # year_discovered = row["YearDiscovered"]
        info = row["info"]

        # Atomic number
        has_the_atomic_number = " has the atomic number of "
        template = name + has_the_atomic_number + "{0}" + "."
        result = perplexity_based_sampling(
            required_param="AtomicNumber",
            current_val=atomic_number,
            template=template,
            df=df,
            model=model,
            tokenizer=tokenizer,
        )
        if result is not False:
            dataset.append((template.format(str(atomic_number)), 1))
            sentence, prob, _, _ = result
            dataset.append((sentence, 0))
            print(sentence)
            print(prob)

        # Symbol
        has_the_symbol = " has the symbol "
        template = name + has_the_symbol + "{0}" + "."
        result = perplexity_based_sampling(
            required_param="Symbol",
            current_val=symbol,
            template=template,
            df=df,
            model=model,
            tokenizer=tokenizer,
        )
        if result is not False:
            dataset.append((template.format(symbol), 1))
            sentence, prob, _, _ = result
            dataset.append((sentence, 0))
            print(sentence)
            print(prob)

        # Standard state
        has_the_standard_state = " appears in its standard state as "
        template = name + has_the_standard_state + "{0}" + "."
        result = perplexity_based_sampling(
            required_param="StandardState",
            current_val=standard_state,
            template=template,
            df=df,
            model=model,
            tokenizer=tokenizer,
        )
        if result is not False:
            dataset.append((template.format(standard_state), 1))
            sentence, prob, _, _ = result
            dataset.append((sentence, 0))
            print(sentence)
            print(prob)

        # # Melting point
        # has_the_melting_point = " has a melting point of "
        # sentence = name + has_the_melting_point + str(melting_point) + " K."
        # dataset.append((sentence, 1))
        #
        # sentence = name + has_the_melting_point + get_other("MeltingPoint", melting_point) + " K."
        # dataset.append((sentence, 0))

        # Group block
        is_in_the_group_block = " is in the "
        template = name + is_in_the_group_block + "{0}" + " group."
        result = perplexity_based_sampling(
            required_param="GroupBlock",
            current_val=group_block,
            template=template,
            df=df,
            model=model,
            tokenizer=tokenizer,
        )
        if result is not False:
            dataset.append((template.format(group_block), 1))
            sentence, prob, _, _ = result
            dataset.append((sentence, 0))
            print(sentence)
            print(prob)
        # # Year discovered
        # sentence = name + " " + get_year_info(year_discovered) + "."
        # dataset.append((sentence, 1))
        #
        # sentence = name + " " + get_year_info(get_other("YearDiscovered", year_discovered)) + "."
        # dataset.append((sentence, 0))

        # Additional info
        template = name + " " + "{0}" + "."
        result = perplexity_based_sampling(
            required_param="info",
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
            "elements_true_false.csv",
        ),
        index=False,
    )  # , header=False)
