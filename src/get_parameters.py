import json as j
import numpy as np
from parsing_json import Functions, Type


def get_prompt_parameters(prompt: str, function_def: Functions) -> tuple[str, list[str], list[Type]]:
    parameters = []
    param_types = []
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


def get_allowed_tokens(variable_type: Type, ai) -> list[int]:
    allowed_digits = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", "\n"]
    allowed_tokens: list[int] = []
    if variable_type.type in ["number", "integer"]:
        for char in "0123456789":
            allowed_digits.append(char)
        allowed_tokens.append(151645)
        if variable_type.type == "number":
            allowed_digits.append('.')
        for item in allowed_digits:
            allowed_tokens.extend(ai.encode(item).tolist()[0])
    return allowed_tokens


def get_parameters(user_prompt: str, function_def: Functions, ai):
    prompt, parameters, param_types = get_prompt_parameters(user_prompt, function_def)
    prompt = ai.encode(prompt).tolist()[0]
    answer = []
    length_param = len(parameters)
    param_index = 0
    index_gen = 0
    answer_list = []

    while param_index < length_param:
        logits: list[float] = ai.get_logits_from_input_ids(prompt)
        logits_cpy = np.array(logits)
        allowed_tokens = get_allowed_tokens(param_types[param_index], ai)

        if len(allowed_tokens) > 0:
            logits_cpy[...] = float('-inf')
            for tokens in allowed_tokens:
                logits_cpy[tokens] = logits[tokens]
        elif param_types[param_index].type == "string":
            encoded = ai.encode("\n").tolist()[0]
            for t in encoded:
                logits_cpy[t] = float('-inf')

        token_index = int(np.argmax(logits_cpy).item())
        prompt.append(token_index)
        answer.append(token_index)

        if ai.decode(token_index).endswith('\n') or ai.decode(token_index) == '"':
            print(f"Found an answer in str, passing through casting {ai.decode(answer)}")

            if param_types[param_index].type == "integer":
                print("INTEGER_TYPE")
                answer_list.append(int(ai.decode(answer).strip("\n").strip(" ").strip("\"")))
            elif param_types[param_index].type == "number":
                print("NUMBER TYPE")
                answer_list.append(float(ai.decode(answer).strip("\n").strip(" ").strip("\"")))
            elif param_types[param_index].type == "string":
                print("STRING TYPE")
                answer_list.append(str(ai.decode(answer).strip("\n").strip(" ").strip("'").strip("\"")))

            param_index += 1
            answer = []
            index_gen = 0

            if param_index < length_param:
                next_param = parameters[param_index]
                next_type = param_types[param_index]
                next_desc = getattr(next_type, "description", "") or ""
                desc_suffix = f" [{next_desc}]" if next_desc else ""

                # FIX: Keep the context inside the same assistant response block
                if next_type.type in ["integer", "number"]:
                    next_prompt_str = f"\n{next_param} ({next_type.type}){desc_suffix}:"
                else:
                    next_prompt_str = f"\n{next_param} ({next_type.type}){desc_suffix}:\""

                # Depending on your tokenizer, you may need to strip leading spaces 
                # to prevent weird token merging, but this matches your current logic.
                prompt += ai.encode(next_prompt_str).tolist()[0]

            index_gen += 1
        if index_gen > 100:
          print("could not find the parameters, exiting the model language")
          break

    print(f"Final list of answers: {answer_list}")
    return (answer_list, parameters)