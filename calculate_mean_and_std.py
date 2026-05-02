import numpy as np

fold_scores = np.array([50, 20, 20, 20], dtype=float)
mean = fold_scores.mean()
std = fold_scores.std(ddof=1)   # <- sample std for folds
print(f"n={len(fold_scores)}  mean={mean:.2f}  std(ddof=1)={std:.2f}")