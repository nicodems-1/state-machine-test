
from parsing_json import definition_parsing, prompt_parsing, Parameters
from llm_sdk import Small_LLM_Model  # type: ignore
import tree_functions as t
import argparse
from get_func_name import match_func
from get_parameters import get_parameters
from building import build_dictionnary
import json as j


def main() -> None:
    """
Execute the main routine for the 'call me maybe' function matching script.

This module serves as the entry point to parse function definitions
and user prompts from JSON files. It validates the function parameters
using Pydantic, initializes a small LLM alongside a FunctionTrie,
and iterates through prompts to match each one to the appropriate function.
Finally, it extracts the required parameters and exports the structured data
to a JSON output file.

Command-line Arguments:
    --input (str): Path to the JSON file containing user prompts.
        Defaults to 'data/input/function_calling_tests.json'.
    --output (str): Directory path for the resulting output files.
        Defaults to 'data/output'.
    --functions_definitions (str): Path to JSON file containing function specs.
        Defaults to 'data/input/functions_definition.json'.

Raises:
    ValueError: If a function definition is successfully parsed but contains
        no parameters (None).
"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="data/input/function_calling_tests.json"
        )
    parser.add_argument("--output", default="data/output")
    parser.add_argument(
        "--functions_definitions",
        default="data/input/functions_definition.json"
        )
    args = parser.parse_args()

    function_definitions = definition_parsing(args.functions_definitions)
    prompts = prompt_parsing(args.input)

    try:
        for item in function_definitions:
            Parameters.model_validate(item.parameters)
            if item.parameters is None:
                print(item.parameters)
                raise ValueError("No parameters provided")
    except ValueError as e:
        print(e)
        exit()
    ai = Small_LLM_Model()
    functree = t.FunctionTrie(ai)

    for function in function_definitions:
        functree.add_function(function.name)
    python_output_list = []

    for prompt in prompts:
        function_typ = match_func(
            prompt.prompt, function_definitions, ai, functree
            )
        function_typstr: str = str(function_typ)
        definition = [
            func for func in function_definitions
            if func.name == function_typstr
            ][0]
        parameters, param_types = get_parameters(prompt.prompt, definition, ai)
        if len(parameters) == 0:
            continue
        output = build_dictionnary(
            prompt.prompt, str(function_typstr), param_types, parameters)
        python_output_list.append(output)
        try: 
            with open("data/output/output.json", "w") as f:
                j.dump(python_output_list, f, indent=4)
                print(j.dumps(output, indent=4))
        except OSError as e:
            print(f"Could not dump file in the ouput file it seems it's protected: {e}")
            exit()


if __name__ == "__main__":
    main()
