import json
import argparse
from pydantic import BaseModel, ValidationError

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="data/input/function_calling_tests.json")
parser.add_argument("--output", default="data/output")
parser.add_argument("--functions_definitions", default="data/input/functions_definition.json")

class Prompting(BaseModel):
    prompt: str

class Type(BaseModel):
    type: str

class Functions(BaseModel):
    name: str
    description: str
    parameters: dict[str, Type] 
    returns: Type

class Ouput(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, type]

all_prompts: list[str] = []
func_def: list[str] = []
func_describe: list[str] = []

try:
    with open (input) as f:
        function_calling = json.load(f)
        for function in function_calling:
            Functions.model_validate(function)
            func_def.append(function['name'])
except ValidationError as e:
    print(e)

try:
    with open (functions_definitions) as f:
        function_calling = json.load(f)
        for prompt_call in function_calling:
            Prompting.model_validate(prompt_call)
            all_prompts.append(prompt_call['prompt'])
            
except ValidationError as e:
    print(e)

