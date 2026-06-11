
from parsing_json import definition_parsing, prompt_parsing
from llm_sdk import Small_LLM_Model
import tree_functions as t
import argparse
from get_func_name import match_func
from get_parameters  import get_parameters



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output")
    parser.add_argument("--functions_definitions", default="data/input/functions_definition.json")
    args = parser.parse_args()    
    
    function_definitions = definition_parsing(args.functions_definitions)
    prompts = prompt_parsing(args.input)

    ai = Small_LLM_Model()
    functree = t.FunctionTrie(ai)

    '''adding functions to the trees'''
    for function in function_definitions:
        functree.add_function(function.name)

    for prompt in prompts:
        print(f"user prompt = {prompt.prompt}")
        function = match_func(prompt.prompt, function_definitions, ai, functree)
        definition = [func for func in function_definitions if func.name == function][0]
        print(f"function_name = {function}")
        print(get_parameters(prompt.prompt, definition, ai))
