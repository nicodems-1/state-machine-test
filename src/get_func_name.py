import numpy as np
import src.tree_functions as t
from src.parsing_json import Functions
from llm_sdk import Small_LLM_Model  # type: ignore


def match_func(call: str, function_def: list[Functions],
               ai: Small_LLM_Model, functree: t.FunctionTrie) -> str:
    tool_box_func: list[str] = []

    for funcs in function_def:
        tool_box_func.append(funcs.name)

    prompt = (f"""List of available functions: {str(tool_box_func).join(", ")}
              These definitions correspond respectively
              to each functions {str(function_def)}
    You should find the corresponding function_name with this: {call}""")

    '''send the prompt to the model with function
    definitions such as Prompt + funcs_names + funcs_def' '''

    answer: list[int] = []
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
        print(ai.decode([token_index]))
        if token_index == 151645:
            break

        current_state = current_state.children[token_index]
        ai_prompt.append(token_index)
        answer.append(token_index)

    answer_txt: str = ai.decode(answer)
    return answer_txt
