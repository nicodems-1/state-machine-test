from src.parsing_json import Functions, Type
from llm_sdk import Small_LLM_Model  # type: ignore
import numpy as np


def get_prompt_parameters(
        prompt: str,
        function_def: Functions) -> tuple[str, list[str], list[Type]]:
    """
    Construct an AI prompt for extracting parameters from a user request.

    Args:
        prompt: The raw user input string.
        function_def: The definition of the target function.

    Returns:
        A tuple containing the formatted AI prompt string, a list of
        parameter names, and a list of parameter types.
    """
    parameters: list[str] = []
    param_types: list[Type] = []
    for parameter, types in function_def.parameters.items():
        parameters.append(parameter)
        param_types.append(types)
    first_param = parameters[0]
    first_type = param_types[0]
    suffix = '"' if first_type.type == "string" else ""

    first_desc = getattr(first_type, "description", "") or ""
    desc_suffix = f" [{first_desc}]" if first_desc else ""

    prompt_ai = (
        f"""<|im_start|>system
You are a strict data extraction engine. {function_def.description}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
{first_param} ({first_type.type}){desc_suffix}:{suffix}""")
    return (prompt_ai, parameters, param_types)


def get_allowed_tokens(
        variable_type: Type, ai: Small_LLM_Model
        ) -> list[int]:
    """
    Determine the allowed LLM token IDs based on the expected variable type.

    Restricts generation by defining a valid subset of tokens (e.g., only
    digits for integers, or digits and periods for floats).

    Args:
        variable_type: The expected data type of the parameter.
        ai: The initialized Small_LLM_Model instance.

    Returns:
        A list of allowed token IDs as integers.
    """
    allowed_digits: list[str] = [
        " 0", " 1", " 2", " 3",
        " 4", " 5", " 6", " 7",
        " 8", " 9", "\n"
                                ]
    allowed_tokens: list[int] = []
    if variable_type.type in ["number", "integer"]:
        for char in "0123456789":
            allowed_digits.append(char)
        allowed_tokens.append(151645)
        if variable_type.type == "number":
            allowed_digits.append('.')
        for item in allowed_digits:
            encoded = ai.encode(item)
            allowed_tokens.extend(encoded[0].tolist())
    return allowed_tokens


def get_parameters(
        user_prompt: str,
        function_def: Functions,
        ai: Small_LLM_Model
        ) -> tuple[list[int | float | str], list[str]]:
    """
    Extract typed function parameters from a user prompt using an LLM.

    Uses logits manipulation to enforce specific type constraints (integers,
    numbers, or strings) during token generation.

    Args:
        user_prompt: The raw user input string.
        function_def: The target function's structure and requirements.
        ai: The initialized Small_LLM_Model instance.

    Returns:
        A tuple containing a list of the extracted parameter values (cast
        to int, float, or str) and a list of their corresponding names.
    """
    prompt_string, parameters, param_types = (
        get_prompt_parameters(user_prompt, function_def)
                                            )
    prompt: list[int] = ai.encode(prompt_string).tolist()[0]
    answer: list[int] = []
    length_param: int = len(parameters)
    param_index: int = 0
    index_gen: int = 0
    answer_list: list[int | float | str] = []

    while param_index < length_param:
        index_gen += 1
        if index_gen > 16:
            print(
                f"\nCould not parameters"
                f"for this prompt : \"{user_prompt}\"")
            print("DISCARDING\n")
            break
        logits: list[float] = ai.get_logits_from_input_ids(prompt)
        logits_cpy = np.array(logits)
        allowed_tokens = get_allowed_tokens(param_types[param_index], ai)

        if len(allowed_tokens) > 0:
            logits_cpy[...] = float('-inf')
            for tokens in allowed_tokens:
                logits_cpy[tokens] = logits[tokens]
        elif param_types[param_index].type == "string":
            encoded: list[int] = ai.encode("\n").tolist()[0]
            for t in encoded:
                logits_cpy[t] = float('-inf')

        token_index = int(np.argmax(logits_cpy).item())
        prompt.append(token_index)
        answer.append(token_index)

        if (ai.decode([token_index]).endswith('\n') or
                ai.decode([token_index]) == '"'):

            if param_types[param_index].type == "integer":
                # print("INTEGER_TYPE")
                answer_list.append(
                    int(ai.decode(answer).strip("\n").strip(" ").strip("\""))
                )
            elif param_types[param_index].type == "number":
                # print("NUMBER TYPE")
                answer_list.append(
                    float(ai.decode(answer).strip("\n").strip(" ").strip("\""))
                    )
            elif param_types[param_index].type == "string":
                # print("STRING TYPE")
                answer_list.append(
                    str(ai.decode(answer).strip("\n").strip("'").strip("\""))
                    )

            param_index += 1
            answer = []
            index_gen = 0

            if param_index < length_param:
                next_param = parameters[param_index]
                next_type = param_types[param_index]
                next_desc = getattr(next_type, "description", "") or ""
                desc_suffix = f" [{next_desc}]" if next_desc else ""

                if next_type.type in ["integer", "number"]:
                    next_prompt_str = (
                        f"\n{next_param} ({next_type.type}){desc_suffix}:"
                        )
                else:
                    next_prompt_str = (
                        f"\n{next_param} ({next_type.type}){desc_suffix}:\""
                        )

                prompt += ai.encode(next_prompt_str).tolist()[0]
    return (answer_list, parameters)
