from typing import Union, Dict, Any


def build_dictionnary(prompt: str, function: str,
                      param_name: list[str], parameters_value: list[Any]
                      ) -> Dict[str, Union[str, Dict[str, Any]]]:
    output_dic: Dict[str, Union[str, Dict[str, Any]]] = {
                                                        "prompt": prompt,
                                                        "name": function,
                                                        "parameters": {}}
    params = output_dic["parameters"]
    assert isinstance(params, dict)
    for i, param in enumerate(parameters_value):
        params[param_name[i]] = param
    return output_dic
