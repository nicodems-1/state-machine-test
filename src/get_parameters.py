import json as j
# from llm_sdk import Small_LLM_Model
import numpy as np
# ai = Small_LLM_Model()
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
    f"""<|im_start|>system Task: Extract the parameters.<|im_end|>
    <|im_start|>user User prompt: {prompt}, Definition is: {function_def.description}
    {first_param} ({first_type.type}): <|im_end|>""")
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
      
"""float model finding"""

def get_parameters(user_prompt:str, function_def: Functions, ai):
    prompt, parameters, param_types = get_prompt_parameters(user_prompt, function_def)

    prompt = ai.encode(prompt).tolist()[0]
    answer = []
    length_param = len(parameters)
    index = 0
    while index < length_param:
        allowed_tokens = get_allowed_tokens(param_types[index], ai)
        # print(ai.decode(allowed_tokens))
        logits: list[float] = ai.get_logits_from_input_ids(prompt)
        logits_cpy = np.array(logits)
        if len(allowed_tokens) != 0:
          logits_cpy[...] = float('-inf')
          for tokens in allowed_tokens:
              logits_cpy[tokens] = logits[tokens]
        token_index = int(np.argmax(logits_cpy).item())
        prompt.append(token_index)
        answer.append(token_index)
        if ai.decode(token_index) == '\n' or ai.decode(token_index) == '"':
            print(f"ai_prompt_decoded = {ai.decode(prompt)}")
            prompt += ai.encode(f' \n{parameters[index]} ({param_types[index].type}):').tolist()[0]
            index += 1
    return (ai.decode(answer))
