import json
from pydantic import BaseModel, ValidationError
from typing import Literal



class Prompting(BaseModel):
    prompt: str

class Type(BaseModel):
    type: Literal["number", "integer", "string"]

class Functions(BaseModel):
    name: str
    description: str
    parameters: dict[str, Type] 
    returns: Type

class Ouput(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, type]


def prompt_parsing(prompt_json: str)-> list[Prompting]:
    all_prompts: list[Prompting] = []
    try:
        with open (prompt_json) as f:
            function_calling = json.load(f)
            for prompt in function_calling:
                all_prompts.append(Prompting.model_validate(prompt))
            return all_prompts
    except ValidationError as e:
        print(e)
        exit()

def definition_parsing(func_def_json: str) -> list[Functions]:
    func_def: list[Functions] = []
    try:
        with open (func_def_json) as f:
            function_calling = json.load(f)
            for definition in function_calling:
                func_def.append(Functions.model_validate(definition))
            return func_def
    except ValidationError as e:
        print(e)
        exit()

