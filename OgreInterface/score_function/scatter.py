import numpy as np


def scatter_add(
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
    shape = list(x.shape)
    shape[dim] = dim_size
    y = np.zeros(shape, dtype=x.dtype)
    np.add.at(y, idx_i, x)

    return y
