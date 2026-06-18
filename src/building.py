from typing import Union, Dict, Any


def build_dictionnary(prompt: str, function: str,
                      param_name: list[str], parameters_value: list[Any]
                      ) -> Dict[str, Union[str, Dict[str, Any]]]:
    """
    Construct a structured dictionary containing the prompt, function, and parameters.

    Args:
        prompt: The raw user input string.
        function: The name of the matched AI function.
        param_name: A list of extracted parameter names.
        parameters_value: A list of the corresponding extracted values.

    Returns:
        A dictionary containing the prompt, function name, and a nested 
        dictionary mapping parameter names to their respective values.
    """
    output_dic: Dict[str, Union[str, Dict[str, Any]]] = {
                                                        "prompt": prompt,
                                                        "name": function,
                                                        "parameters": {}}
    params = output_dic["parameters"]
    assert isinstance(params, dict)
    for i, param in enumerate(parameters_value):
        params[param_name[i]] = param
    return output_dic
