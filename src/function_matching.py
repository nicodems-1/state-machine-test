from json import load, dump
import numpy as np
import tree_functions as t


'''parsing to protect'''
with open("functions_definition.json") as f:
    function_def:list[dict] = load(f)

with open("functions_calling_test.json") as f:
    calls = load(f)
'''parsing to protect'''


"""model instanciation"""
ai = Small_LLM_Model()

dic_list = []
functree = t.FunctionTrie(ai)

available = ""
defs = ""
function_dic = {}
for funcs in function_def:

    defs = ", " + str(funcs["description"]) + "\n"
    defs = defs.strip(',').strip()

    available = ", " + "\"" + str(funcs["name"]) + "\"" + "\n"
    available = available.strip(',').strip()

    functree.add_function(available)
    function_dic.update({available: defs})

funcs = str(function_dic)

for element in calls:
    call_prompt = element['prompt']
    prompt = (f"This is a dictionnary, the format is: \"function_name\":\"function_description\" : {funcs}\n"
            f"You should find the corresponding function_name with this: {call_prompt}")

    # print(call_prompt)

    '''send the prompt to the model with function definitions such as Prompt + funcs_names + funcs_def' '''

    answer = []
    ai_prompt = ai.encode(prompt).tolist()[0]
    current_state = functree.root


    while True:
        logits: list[float] = ai.get_logits_from_input_ids(ai_prompt)
        logits_cpy = np.array(logits)
        logits_cpy[...] = float('-inf')

        allowed_tokens = functree.get_valid_token(current_state)
        for tokens in allowed_tokens:
            logits_cpy[tokens] = logits[tokens]
        token_index = int(np.argmax(logits_cpy).item())
        if token_index == 151645:
            break
        current_state = current_state.children[token_index]
        ai_prompt.append(token_index)
        answer.append(token_index)
    # print(ai.decode(answer))
    # print('++++++++++++++++++++++')


    '''Apply the masking logic, once the func is gotten, find the matching definition thanks to method and start building the dictionnary'''

    answer = ai.decode(answer)
    answer = answer.strip('\"')

    """ looking for the function match in the dic list"""

    for func_def in function_def:
        if func_def["name"] == answer:
            function_choosen = func_def
    

    def get_parameters_names():
        params = []
        for element in function_choosen['parameters']:
            params.append(element)
        return params



    def building_dictionnary(): 
        output = {"prompt":call_prompt, "name": answer}
        parameters = get_parameters_names()
        param_dic = {}
        default = 0
        for param in parameters:
            param_dic.update({param: default})
        output.update({"parameters": param_dic})
        return output
    
    def update_type(output = dict):
        types = []
        output = building_dictionnary()
        # print(output["parameters"])


    output = building_dictionnary()
    update_type(output)
    dic_list.append(output)
    '''Dictionnary is built, fill it with right data type thanks to the function signature'''
    

    '''Last step : fill out the output json sheet with the right output'''
    print(output)
    with open("output.json", "w") as f:
        print(dump(dic_list, f, indent=4))