import json
from pydantic import BaseModel, ValidationError, model_validator
from typing import Literal, Self


class Prompting(BaseModel):
    prompt: str

    @model_validator(mode="after")
    def check_empty_field(self) -> Self:
        for field, name in self:
            print(f"Field_Type == {field}")
            print(f"Name_Type == {name}")
            if name in ("", [], {}) or field in ("", [], {}):
                raise ValueError(f"Field {field}, cannot be empty")
        return self


class Type(BaseModel):
    type: Literal["number", "integer", "string"]


class Parameters(BaseModel):
    dict[str, Type]


class Functions(BaseModel):
    name: str
    description: str
    parameters: dict[str, Type]
    returns: Type

    @model_validator(mode="after")
    def check_empty_field(self) -> Self:
        for field, name in self:
            print(f"Field_Function == {field}")
            print(f"Name_Functions == {name}")
            print(f"SELF == {self}")
            if name in ("", [], {}) or field in ("", [], {}):
                raise ValueError(f"Field {field}, cannot be empty")
        return self


class Output(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, type]


def prompt_parsing(prompt_json: str) -> list[Prompting]:
    all_prompts: list[Prompting] = []
    try:
        with open(prompt_json) as f:
            function_calling = json.load(f)
            for prompt in function_calling:
                all_prompts.append(Prompting.model_validate(prompt))
            return all_prompts
    except ValidationError as e:
        print(f"Parsing Error: function_calling: {e}")
        exit()


def definition_parsing(func_def_json: str) -> list[Functions]:
    func_def: list[Functions] = []
    try:
        with open(func_def_json) as f:
            function_calling = json.load(f)
            for definition in function_calling:
                func_def.append(Functions.model_validate(definition))
            return func_def
    except ValidationError as e:
        print(f"Parsing Error: function_defintions: {e}")
        exit()
