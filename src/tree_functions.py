class TrieNode():
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class FunctionTrie():
    def __init__(self, ai_instance):
        self.root = TrieNode()
        self.eos = 151645
        self.ai = ai_instance

    def add_function(self, function_name):
        self.tokenizer = self.ai.encode(function_name).tolist()[0]
        # print(self.tokenizer)
        current_node = self.root
        for token_id in self.tokenizer:
            if token_id in current_node.children:
                current_node = current_node.children[token_id]
            else:
                new_node = TrieNode()
                current_node.children[token_id] = new_node
                current_node = new_node
        current_node.is_end_of_word = True

    def get_valid_token(self, current_node):
        allowed_tokens = list(current_node.children.keys())

        if current_node.is_end_of_word:
            allowed_tokens.append(self.eos)
        return allowed_tokens