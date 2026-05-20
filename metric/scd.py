"""SCD --- Sum of Correlations of Differences (Aslantas & Bendes 2015).

SCD = corr(F - A, B) + corr(F - B, A)

trong đó corr là Pearson correlation. Cao tốt hơn.
"""
import numpy as np


def _pearson(x, y):
    x = x.flatten().astype(np.float64)
    y = y.flatten().astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    num = float((x * y).sum())
    den = float(np.sqrt((x * x).sum() * (y * y).sum())) + 1e-12
    return num / den


def scd(A, B, F):
    """Sum of Correlations of Differences.

    Input: 2D grayscale arrays (uint8 or float).
    Output: scalar in roughly [-2, +2], higher better.
    """
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    F = F.astype(np.float64)
    d1 = F - A
    d2 = F - B
    return _pearson(d1, B) + _pearson(d2, A)
