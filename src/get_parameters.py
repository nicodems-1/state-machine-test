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
  prompt_ai = (
    f"""User prompt: {prompt}.
    For context, definition is {function_def.description}
    system Task: You are a strict data extraction engine
    {first_param} ({first_type.type}):""")
  return (prompt_ai, parameters, param_types)

def get_allowed_tokens(variable_type: Type, ai) -> list[int]:
    allowed_digits = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7",  " 8", " 9", "\n"]
    allowed_tokens: list[int] = []
    if variable_type.type in ["number", "integer"]:
        for char in "0123456789":
          allowed_digits.append(char)
        allowed_tokens.append(151645)
        if variable_type == "number":
          allowed_digits.append('.')
        for item in allowed_digits:
          allowed_tokens.extend(ai.encode(item).tolist()[0])
    return allowed_tokens
      

def get_parameters(user_prompt:str, function_def: Functions, ai):
    prompt, parameters, param_types = get_prompt_parameters(user_prompt, function_def)
    prompt = ai.encode(prompt).tolist()[0]



    answer = []
    length_param = len(parameters)
    param_index = 0
    index_gen = 0
    string_mode = False
    answer_list = []


    while param_index < length_param:
        logits: list[float] = ai.get_logits_from_input_ids(prompt)
        logits_cpy = np.array(logits)
        allowed_tokens = get_allowed_tokens(param_types[param_index], ai)

        if len(allowed_tokens) > 0:
          logits_cpy[...] = float('-inf')
          for tokens in allowed_tokens:
              logits_cpy[tokens] = logits[tokens]

        token_index = int(np.argmax(logits_cpy).item())

        prompt.append(token_index)
        answer.append(token_index)
        if ai.decode(token_index).endswith('\n') or ai.decode(token_index) == '"':
            if param_types[param_index].type in ["integer", "number"]:
              prompt += ai.encode(f' \n{parameters[param_index]} ({param_types[param_index].type}):').tolist()[0]
            else:
              prompt += ai.encode(f'{parameters[param_index]} ({param_types[param_index].type}):\"').tolist()[0]
            string_mode = False
            index_gen = 0
            if param_index < len(param_types):
              print(f"Found an answer in str, passing through casting {ai.decode(answer)}")
              if param_types[param_index].type == "integer":
                print("INTEGER_TYPE")
                answer_list.append(int(ai.decode(answer).strip("\n").strip(" ")))
              elif param_types[param_index].type == "number":
                print("NUMBER TYPE")
                answer_list.append(float(ai.decode(answer).strip("\n").strip(" ")))
              elif param_types[param_index].type == "string":
                print("STRING TYPE")
                answer_list.append(str(ai.decode(answer).strip("\n").strip(" ")))
            param_index += 1
            answer = []
        index_gen += 1
        if index_gen > 100:
           print("could not find the parameters, exiting the model language")
           break
    print(f"Final list of answers: {answer_list}")
    return (answer_list, parameters)
