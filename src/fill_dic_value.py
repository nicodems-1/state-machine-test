from llm_sdk import Small_LLM_Model
import numpy as np
ai = Small_LLM_Model()

def prompt_factory():
    yes = {'prompt': 'What is the sum of 2 and 3?', 'name': 'fn_add_numbers', 'parameters': {'a': {"type": "number"}, 'b': {"type": "number"}}}
    variables = []
    little_dic = yes['parameters']
    for element in yes['parameters']:
        variables.append(element)
    for keys in variables:
        print(little_dic[keys])

    prompt = (f"Give me the parameter \"a\" coresponding to this prompt: {str(yes['prompt'])}, the parameter type is Number"
    f"this is the function definition dictionnary : {str(yes)}")
    return(prompt)


def model_finding(prompt:str):
    print(prompt)
    allowed_tokens = ai.encode("1234567890").tolist()[0]
    allowed_tokens.append(151645)
    allowed_tokens.append(1)
    print(allowed_tokens)
    answer = []
    prompt = ai.encode(prompt).tolist()[0]
    while True:
        logits: list[float] = ai.get_logits_from_input_ids(prompt)
        logits_cpy = np.array(logits)
        logits_cpy[...] = float('-inf')

        for tokens in allowed_tokens:
            logits_cpy[tokens] = logits[tokens]
        token_index = int(np.argmax(logits_cpy).item())
        if token_index == 151645:
            break
        prompt.append(token_index)
        answer.append(token_index)
        print(ai.decode(answer))
    print('++++++++++++++++++++++')


prompt = prompt_factory()
model_finding(prompt)