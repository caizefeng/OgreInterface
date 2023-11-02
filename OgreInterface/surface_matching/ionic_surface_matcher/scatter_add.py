import numpy as np


def scatter_add_bin(
    x: np.ndarray,
    idx_i: np.ndarray,
    dim_size: int,
    dim: int = 0,
) -> np.ndarray:
    """
    Sum over values with the same indices.

    Args:
        x: input values
        idx_i: index of center atom i
        dim_size: size of the dimension after reduction
        dim: the dimension to reduce

    Returns:
        reduced input
    """
    y = np.zeros(dim_size)
    tmp = np.bincount(idx_i, weights=x)
    y[np.arange(len(tmp)).astype(int)] = tmp

    return y
