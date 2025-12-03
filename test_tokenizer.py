from code_mdl import tokenizer
import numpy as np

def test_tokenizer():
    text = "Hello, world!"
    tokens = tokenizer(text)
    print(f"Text: '{text}'")
    print(f"Tokens: {tokens}")
    
    assert isinstance(tokens, list), "Tokenizer should return a list"
    assert all(isinstance(t, int) for t in tokens), "Tokens should be integers"
    assert len(tokens) > 0, "Should return tokens for non-empty string"
    
    print("Tokenizer test passed!")

if __name__ == "__main__":
    test_tokenizer()
