*This project has been created as part of the 42 curriculum by niverdie*

## Description
The project goal is to generate json format output from two input files.  
One file is a list of prompt.  
One file is a list of Functions definitions.  
The extract engine is a small local model (qwen 0.6B for me).  
The problem with LLM is that they tend to talk a lot and
generally struggle to generate Json format.  
Constrain decoding makes the extraction operation more efficient and less prone to errors.

## Prerequisite
Hardware: You'll need 5 gigas to store the local model.  
Environment: All the command are executed inside a virtual environment managed with UV.  

## Instructions
Installation : make install  
Debugging : make debug  
Linting : make lint  

## Resources
ChatML: Utilized official Qwen documentation to implement ChatML tags, significantly improving the LLM's adherence to the prompt structure.  
Gemini: Consulted for optimizing prompt engineering techniques for small-scale language models.   

## Strategy
The pipeline processes input prompts against a set of function definitions.  
The core logic relies on Logit Masking (Constrained Decoding) to force the model to select only from valid tokens at every step of the generation process. 



**The main:**  
```c
    for prompt in prompts:
        function_typ = match_func(
            prompt.prompt, function_definitions, ai, functree
            )
        function_typstr: str = str(function_typ)
        definition = [
            func for func in function_definitions
            if func.name == function_typstr
            ][0]
        parameters, param_types = get_parameters(prompt.prompt, definition, ai)
        output = build_dictionnary(
            prompt.prompt, str(function_typstr), param_types, parameters)
        python_output_list.append(output)
        with open("data/output/output.json", "w") as f:
            j.dump(python_output_list, f, indent=4)
            print(j.dumps(output, indent=4))
```


### <ins>Finding Function name:</ins>  

1. Validation: Both definitions.json and prompt.json are validated using Pydantic.  

2. Indexing: Function names are tokenized and organized into a tree structure (functree).  

3. Function Matching: The LLM generates the function name. At each step, the model's logits are masked to allow only tokens that lead to a valid function name.  

4. Parameter Extraction: Once the function is identified, the pipeline continues the prompt with parameter names/types to guide the LLM's extraction.  



**The function matching algorithm:**  
```c
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
        answer.append(token_index) ```
```

The functions extracting parameters works on the same principle.  
One difference is that it is completing the prompt during execution with the pamarameter name and type.  
By completing the prompt with the extracted parameter, the model gain more context for the next extraction.  
For integer and number we apply the logit masking technic

## Output
**The final output is build in a python dictionnary, then dumped with json.load.**  

```c
def build_dictionnary(prompt, function, param_name , parameters_value):
    output_dic:  = {
                    "prompt": prompt,
                    "name": function,
                    "parameters": {}
                    }
    params = output_dic["parameters"]
    for i, param in enumerate(parameters_value):
        params[param_name[i]] = param
    return output_dic
```