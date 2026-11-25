"""monitor module - TODO"""
import numpy as np


def psi(a, b, bins=10):
    a = np.asarray(a)
    b = np.asarray(b)

    edges = np.quantile(a, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    edges[0] = -np.inf
    edges[-1] = np.inf

    pa = np.histogram(a, bins=edges)[0] / len(a)
    pb = np.histogram(b, bins=edges)[0] / len(b)

    eps = 1e-6
    pa = pa + eps
    pb = pb + eps

    return np.sum((pb - pa) * np.log(pb / pa))