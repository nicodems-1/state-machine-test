def build_dictionnary(prompt: str, function: str,param_name, parameters_value):
    output_dic = {}
    output_dic.update({"prompt":prompt})
    output_dic.update({"name": function})
    output_dic.update({"parameters": {}})
    for i, param in enumerate(parameters_value):
        output_dic['parameters'].update({param_name[i]: param})
    return output_dic