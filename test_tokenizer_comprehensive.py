"""
Comprehensive test for the BPE tokenizer implementation.
"""
from main import tokenizer, caption_complexity
import numpy as np

def test_tokenizer():
    """Test basic tokenizer functionality."""
    print("=" * 50)
    print("TEST 1: Basic Tokenizer Functionality")
    print("=" * 50)
    
    # Test 1: Basic tokenization
    text1 = "Hello, world!"
    tokens1 = tokenizer(text1)
    print(f"Text: '{text1}'")
    print(f"Tokens: {tokens1}")
    print(f"Token count: {len(tokens1)}")
    
    assert isinstance(tokens1, list), "Tokenizer should return a list"
    assert all(isinstance(t, int) for t in tokens1), "Tokens should be integers"
    assert len(tokens1) > 0, "Should return tokens for non-empty string"
    print("✓ Basic tokenization passed\n")
    
    # Test 2: Different text lengths
    text2 = "A"
    tokens2 = tokenizer(text2)
    print(f"Text: '{text2}'")
    print(f"Tokens: {tokens2}")
    print(f"Token count: {len(tokens2)}")
    assert len(tokens2) >= 1, "Single character should produce at least one token"
    print("✓ Single character tokenization passed\n")
    
    # Test 3: Empty string
    text3 = ""
    tokens3 = tokenizer(text3)
    print(f"Text: '{text3}'")
    print(f"Tokens: {tokens3}")
    print(f"Token count: {len(tokens3)}")
    assert len(tokens3) == 0, "Empty string should produce no tokens"
    print("✓ Empty string tokenization passed\n")
    
    # Test 4: Longer text
    text4 = "This is a longer sentence with multiple words to test the tokenizer."
    tokens4 = tokenizer(text4)
    print(f"Text: '{text4}'")
    print(f"Tokens: {tokens4}")
    print(f"Token count: {len(tokens4)}")
    assert len(tokens4) > len(tokens1), "Longer text should produce more tokens"
    print("✓ Long text tokenization passed\n")

def test_caption_complexity():
    """Test caption complexity calculation with BPE tokenizer."""
    print("=" * 50)
    print("TEST 2: Caption Complexity Calculation")
    print("=" * 50)
    
    # Mock data
    c = "A cat sitting on a mat"
    prompt_set = []  # Empty prompt set
    prompt_embeddings = np.zeros((0, 384))
    
    # Test with empty prompt set (only length penalty)
    alpha = 0.01
    score = caption_complexity(c, prompt_set, prompt_embeddings, alpha=alpha, beta=0.1)
    tokens = tokenizer(c)
    expected_score = alpha * len(tokens)
    
    print(f"Caption: '{c}'")
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}")
    print(f"Caption complexity score: {score}")
    print(f"Expected score (alpha * token_count): {expected_score}")
    
    assert abs(score - expected_score) < 1e-6, f"Score {score} != Expected {expected_score}"
    print("✓ Caption complexity calculation passed\n")

def test_tokenizer_consistency():
    """Test that tokenizer produces consistent results."""
    print("=" * 50)
    print("TEST 3: Tokenizer Consistency")
    print("=" * 50)
    
    text = "Consistent tokenization test"
    tokens1 = tokenizer(text)
    tokens2 = tokenizer(text)
    
    print(f"Text: '{text}'")
    print(f"First call: {tokens1}")
    print(f"Second call: {tokens2}")
    
    assert tokens1 == tokens2, "Tokenizer should produce consistent results"
    print("✓ Tokenizer consistency passed\n")

if __name__ == "__main__":
    try:
        test_tokenizer()
        test_caption_complexity()
        test_tokenizer_consistency()
        
        print("=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("=" * 50)
        print("\nThe BPE tokenizer using tiktoken is working correctly!")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
