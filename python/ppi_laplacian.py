import numpy as np
from scipy.sparse import issparse

if __package__:
    from .shared import NeuroFANNError
else:
    from shared import NeuroFANNError


def ppi_laplacian(adjacency):
    try:
        array = np.asarray(adjacency.toarray() if issparse(adjacency) else adjacency)
    except (TypeError, ValueError) as exception:
        raise NeuroFANNError("NeuroFANN:InvalidNetwork", "The PPI weight matrix must be numeric.") from exception
    if array.ndim != 2 or array.size == 0 or array.shape[0] != array.shape[1] or array.dtype.kind not in "biuf":
        raise NeuroFANNError("NeuroFANN:InvalidNetwork", "The PPI weight matrix must be a nonempty real square matrix.")
    weights = np.array(array, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise NeuroFANNError("NeuroFANN:InvalidNetwork", "PPI weights must be finite and nonnegative.")
    if not np.array_equal(weights, weights.T):
        raise NeuroFANNError("NeuroFANN:InvalidNetwork", "The PPI weight matrix must be exactly symmetric.")
    count = weights.shape[0]
    degree = np.zeros(count, dtype=np.float64)
    with np.errstate(over="ignore"):
        for column in range(count):
            degree = degree + weights[:, column]
    if not np.all(np.isfinite(degree)):
        raise NeuroFANNError("NeuroFANN:InvalidNetwork", "PPI weighted degrees exceed the floating-point range.")
    scale = np.zeros(count, dtype=np.float64)
    connected = degree > 0
    scale[connected] = 1.0 / np.sqrt(degree[connected])
    return np.eye(count) - (scale[:, None] * weights) * scale[None, :]
