from src.parsing_json import function_calling_parsing, function_definition_parsing
from llm_sdk import Small_LLM_Model

function_definitions = function_definition_parsing()
prompt = function_calling_parsing()

ai = Small_LLM_Model()

for function in 
