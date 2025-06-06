import pandas as pd
import os
from datagen_perplexity.utils import get_project_root
from datagen_perplexity.utils import get_llm, perplexity_based_sampling
from datagen_perplexity.utils import enable_determinism
import tqdm


def main(llm_name, model, tokenizer):
    # Read the CSV file
    df = pd.read_csv(os.path.join(get_project_root(), "resources", "animals.csv"))

    dataset = []
    for i, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        # Animal, Class, Habitat, Diet, Locomotion, Human Uses, Info
        animal_name = row["Animal"].lower()
        animal_class = row["Class"].lower()
        habitat = row["Habitat"].lower()
        diet = row["Diet"].lower()
        locomotion = row["Locomotion"].lower()
        human_uses = row["Human Uses"].lower()
        info = row["Info"].lower()

        # Animal class
        is_a = " is a "
        template = "The " + animal_name + is_a + "{0}" + "."
        result = perplexity_based_sampling(
            required_param="Class",
            current_val=animal_class,
            template=template,
            df=df,
            model=model,
            tokenizer=tokenizer,
            lower=True,
        )
        if result is not False:
            dataset.append((template.format(animal_class), 1))
            print("True: " + template.format(animal_class))
            sentence, prob, _, _ = result
            dataset.append((sentence, 0))
            print("False: " + sentence)
            print(prob)

        # Habitat
        has_habitat = " has a habitat of "
        template = "The " + animal_name + has_habitat + "{0}" + "."
        result = perplexity_based_sampling(
            required_param="Habitat",
            current_val=habitat,
            template=template,
            df=df,
            model=model,
            tokenizer=tokenizer,
            lower=True,
        )
        if result is not False:
            dataset.append((template.format(habitat), 1))
            sentence, prob, _, _ = result
            dataset.append((sentence, 0))
            print(sentence)
            print(prob)

        # Diet
        has_diet = " has a diet of "
        template = "The " + animal_name + has_diet + "{0}" + "."
        result = perplexity_based_sampling(
            required_param="Diet",
            current_val=diet,
            template=template,
            df=df,
            model=model,
            tokenizer=tokenizer,
            lower=True,
        )
        if result is not False:
            dataset.append((template.format(diet), 1))
            sentence, prob, _, _ = result
            dataset.append((sentence, 0))
            print(sentence)
            print(prob)

        # Locomotion
        uses_locomotion = " uses "
        template = "The " + animal_name + uses_locomotion + "{0}" + " for locomotion."
        result = perplexity_based_sampling(
            required_param="Locomotion",
            current_val=locomotion,
            template=template,
            df=df,
            model=model,
            tokenizer=tokenizer,
            lower=True,
        )

        if result is not False:
            dataset.append((template.format(locomotion), 1))
            sentence, prob, _, _ = result
            dataset.append((sentence, 0))
            print(sentence)
            print(prob)

        human_uses_for = "Human uses for "
        template = human_uses_for + animal_name + " include " + "{0}" + "."
        result = perplexity_based_sampling(
            required_param="Human Uses",
            current_val=human_uses,
            template=template,
            df=df,
            model=model,
            tokenizer=tokenizer,
            special_cases={
                "none": "There are no specific human uses for " + animal_name + "."
            },
        )

        if result is not False:
            # Human Uses
            if human_uses != "none":
                human_uses_for = "Human uses for "
                true_sentence = (
                    human_uses_for + animal_name + " include " + human_uses + "."
                )
            else:
                true_sentence = (
                    "There are no specific human uses for " + animal_name + "."
                )
            dataset.append((true_sentence, 1))

            sentence, prob, _, _ = result
            dataset.append((sentence, 0))
            print(sentence)
            print(prob)

        template = "The " + animal_name + " " + "{0}" + "."
        result = perplexity_based_sampling(
            required_param="Info",
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
            "animals_true_false.csv",
        ),
        index=False,
    )  # , header=False)
