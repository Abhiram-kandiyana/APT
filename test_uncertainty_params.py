from code_mdl import get_stochastic_params

def test_stochastic_params():
    K = 10
    params = get_stochastic_params(K)
    
    print(f"Generated {len(params)} params for K={K}:")
    for t, p in params:
        print(f"T={t}, P={p}")
        
    assert len(params) == K
    assert len(set(params)) == K, "Pairs must be unique"
    
    # Check ranges
    for t, p in params:
        assert 0.5 <= t <= 1.5
        assert 0.7 <= p <= 1.0
        
    print("Test passed!")

if __name__ == "__main__":
    test_stochastic_params()
