from llm_sdk import Small_LLM_Model  # type: ignore


class TrieNode():
    def __init__(self) -> None:
        """Represent a single node within the token-based FunctionTrie."""
        self.children: dict = {}
        self.is_end_of_word = False


class FunctionTrie():
    """Manage a Trie structure of LLM tokens to validate function names."""
    def __init__(self, ai_instance: Small_LLM_Model):
        self.root = TrieNode()
        self.eos = 151645
        self.ai: Small_LLM_Model = ai_instance

    def add_function(self, function_name: str) -> None:
        """
        Encode a function name into tokens and insert it into the Trie.

        Args:
            function_name: The target function's name as a string.
        """
        self.tokens = self.ai.encode(function_name).tolist()[0]
        current_node = self.root
        for token_id in self.tokens:
            if token_id in current_node.children:
                current_node = current_node.children[token_id]
            else:
                new_node = TrieNode()
                current_node.children[token_id] = new_node
                current_node = new_node
        current_node.is_end_of_word = True

    def get_valid_token(self, current_node: TrieNode) -> list[int]:
        """
        Retrieve all permissible next token IDs from a given Trie node.

        Args:
            current_node: The node currently being evaluated.

        Returns:
            A list of allowed token IDs, including the End-Of-Sequence (EOS)
            token if the current node completes a valid function name.
        """
        allowed_tokens = list(current_node.children.keys())

        if current_node.is_end_of_word:
            allowed_tokens.append(self.eos)
        return allowed_tokens
