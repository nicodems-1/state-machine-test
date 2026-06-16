
from parsing_json import definition_parsing, prompt_parsing
from llm_sdk import Small_LLM_Model  # type: ignore
import tree_functions as t
import argparse
from get_func_name import match_func
from get_parameters import get_parameters
from building import build_dictionnary
import json as j

if __name__ == "__main__":
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

    ai = Small_LLM_Model()
    functree = t.FunctionTrie(ai)

    '''adding functions to the trees'''
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
        output = build_dictionnary(
            prompt.prompt, str(function_typstr), param_types, parameters)
        python_output_list.append(output)
        with open("data/output/output.json", "w") as f:
            j.dump(python_output_list, f, indent=4)
            print(j.dumps(output, indent=4))
