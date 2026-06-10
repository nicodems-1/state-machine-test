import json as j
# from llm_sdk import Small_LLM_Model
import numpy as np
# ai = Small_LLM_Model()

'''parsing to protect'''
with open("functions_definition.json") as f:
    function_def:list[dict] = j.load(f)

def get_parameter_len(function):
  return (len(function['parameters']))

"""float model finding"""
# def model_finding(prompt:str, function):
#     print(prompt)
#     allowed_items = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7",  " 8", " 9", "\n", "."]
#     allowed_tokens = []
#     for char in "0123456789":
#       allowed_items.append(char)
#     allowed_tokens.append(151645)
    
#     for item in allowed_items:
#       allowed_tokens.extend(ai.encode(item).tolist()[0])
#     prompt = ai.encode(prompt).tolist()[0]
#     answer = []
#     length_param = get_parameter_len(function)
#     while length_param != 0:
#         logits: list[float] = ai.get_logits_from_input_ids(prompt)
#         logits_cpy = np.array(logits)
#         logits_cpy[...] = float('-inf')

#         for tokens in allowed_tokens:
#             logits_cpy[tokens] = logits[tokens]
#         token_index = int(np.argmax(logits_cpy).item())
#         prompt.append(token_index)
#         answer.append(token_index)
#         parameters = ai.decode(answer)
#         if ai.decode(token_index) == '\n':
#             length_param -= 1
#             prompt += ai.encode(' \nb:').tolist()[0]
#     print(parameters.split())


function = function_def[4]
parameters = []
for parameter in function['parameters']:
  parameters.append(parameter)

print(parameters)

types = []
for param in parameters:
  types.append(function['parameters'][param]['type'])
print(types)
first_param = parameters[0]
first_type = types[0]
call = "what is the sum of 45 and 98"
prompt = f"""User prompt: {call}
Task: Extract the parameters.
{first_param} ({types[0]}):"""
print(prompt)
model_finding(prompt, function)