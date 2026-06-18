import json
from pydantic import BaseModel, ValidationError, model_validator
from typing import Literal, Self


class Prompting(BaseModel):
    """Represent a user prompt for function calling."""
    prompt: str

    @model_validator(mode="after")
    def check_empty_field(self) -> Self:
        """Ensure no fields within the prompt model are empty."""
        for field, name in self:
            if name in ("", [], {}) or field in ("", [], {}):
                raise ValueError(f"Field {field}, cannot be empty")
        return self


class Type(BaseModel):
    """Define the allowed data types for function parameters."""
    type: Literal["number", "integer", "string"]


class Parameters(BaseModel):
    """Represent a collection of parameter names
    mapped to their allowed types."""
    dict[str, Type]


class Functions(BaseModel):
    """Define the schema, metadata, and parameters
    for an available AI function."""
    name: str
    description: str
    parameters: dict[str, Type]
    returns: Type

    @model_validator(mode="after")
    def check_empty_field(self) -> Self:
        """Ensure no fields within the function definition are empty."""
        parameter = self.parameters
        for keys in parameter:
            if not keys:
                raise ValueError("The parameter must have a Name: ")
        for field, name in self:
            if name in ("", [], {}) or field in ("", [], {}):
                raise ValueError(f"Field {field}, cannot be empty")
        return self


class Output(BaseModel):
    """Represent the final structured output mapping a prompt
    to a function and its arguments."""
    prompt: str
    name: str
    parameters: dict[str, type]


def prompt_parsing(prompt_json: str) -> list[Prompting]:
    """
    Parse a JSON file of user prompts into a list of Prompting models.

    Args:
        prompt_json: The file path to the JSON containing user prompts.

    Returns:
        A list of validated Prompting objects.
    """
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
    except json.JSONDecodeError as e:
        print(f"The function_definitions file is not correct : {e}")
        exit()


def definition_parsing(func_def_json: str) -> list[Functions]:
    """
    Parse a JSON file of function specifications
    into a list of Functions models.

    Args:
        func_def_json: The file path to the JSON containing
        function definitions.

    Returns:
        A list of validated Functions objects.
    """
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
    except json.JSONDecodeError as e:
        print(f"The function_definitions file is not correct : {e}")
        exit()
