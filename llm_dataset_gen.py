from datagen_perplexity.utils import get_llm
from datagen_perplexity.utils import enable_determinism
import importlib



LLMS = [
    "meta-llama/Meta-Llama-3-8B",
    #"meta-llama/Llama-2-7b-hf",
    #"facebook/opt-6.7b"
]


DATASETS = [
    "animals",
    "capitals",
    "companies",
    "elements",
    "inventions"
]

#DATASETS = ["color"]

def execute_main_function(*args, **kwargs):
    """
    Dynamically import and execute the main function from a script.

    Parameters:
        dataset (str): The name of the dataset to construct the script name.
    """
    try:
        # Construct module name dynamically
        module_name = f"datagen_perplexity.llm_dataset_{dataset}_gen"

        # Dynamically import the module
        module = importlib.import_module(module_name)

        # Execute the main function
        if hasattr(module, "main"):
            module.main(*args, **kwargs)
        else:
            print(f"The module '{module_name}' does not have a 'main' function.")
    except ModuleNotFoundError:
        print(f"Module '{module_name}' not found.")
    except Exception as e:
        print(f"An error occurred while executing '{module_name}': {e}")


if __name__ == "__main__":
    for llm in LLMS:
        model, tokenizer = get_llm(llm)
        for dataset in DATASETS:
            enable_determinism()
            print(f"Processing dataset: {dataset}")
            execute_main_function(llm, model, tokenizer)
