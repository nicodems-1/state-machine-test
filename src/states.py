class State:
    """
    We define a state object which provides some utility functions for the
    individual states within the state machine.
    """

    def __init__(self):
        print (f'Processing current state:{self}')

    def on_event(self, event):
        """
        Handle events that are delegated to this State.
        """
        return self

    def __repr__(self):
        """
        Leverages the __str__ method to describe the State.
        """
        return self.__str__()

    def __str__(self):
        """
        Returns the name of the State.
        """
        return self.__class__.__name__
    
    def get_allowed_tokens(self):
        """
        Logits masking strategy llm is choosing from specified alphabet
        """
        pass

class InitState(State):

    def get_allowed_tokens(self):
        return "{"
    
    def on_event(self, event):
        return FunctionKeyState()
    
class FunctionKeyState(State):
    
    def get_allowed_tokens(self):
        return "function"
    
    def on_event(self, event):
        return ColonState()
    
class ColonState(State):

    def get_allowed_tokens(self):
        return ":"
    
    def on_event(self, event):
        return FunctionValueState()

class FunctionValueState(State):

    def get_allowed_tokens(self):
        return "add_numbers"
    
    def on_event(self, event):
        return CommaState()

class CommaState(State):

    def get_allowed_tokens(self):
        return ","
    
    def on_event(self, event):
        return ArgumentKeyState()
    
class ArgumentKeyState(State):

    def get_allowed_tokens(self):
        return "arguments"
    
    def on_event(self, event):
        return ColonState()
    