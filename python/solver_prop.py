from collections.abc import Mapping

import numpy as np
from scipy import sparse
from scipy.linalg.lapack import dgetrs

if __package__:
    from .shared import NeuroFANNError
else:
    from shared import NeuroFANNError


def solver_prop(factor, right_hand_side, transpose_system=False):
    if not isinstance(factor, Mapping) or not {"LU", "Pivot"}.issubset(factor):
        raise NeuroFANNError("NeuroFANN:InvalidPropagationFactor", "Create the propagation factor with factor_prop.")
    lu = factor["LU"]
    raw = right_hand_side.toarray() if sparse.issparse(right_hand_side) else np.asarray(right_hand_side)
    if (
        raw.ndim not in (1, 2)
        or raw.shape[0] != lu.shape[0]
        or raw.dtype.kind not in "biuf"
        or not np.all(np.isfinite(raw))
    ):
        raise NeuroFANNError(
            "NeuroFANN:NonfinitePropagation",
            "The propagation right-hand side must have one finite real row per protein.",
        )
    rhs = np.asarray(raw, dtype=np.float64)
    if rhs.size == 0:
        return np.empty(rhs.shape, dtype=np.float64)
    solution, info = dgetrs(lu, factor["Pivot"], rhs, trans=1 if transpose_system else 0, overwrite_b=0)
    if info != 0:
        raise NeuroFANNError("NeuroFANN:NonfinitePropagation", "The propagation solve failed.")
    if not np.all(np.isfinite(solution)):
        raise NeuroFANNError("NeuroFANN:NonfinitePropagation", "The propagation solve produced nonfinite values.")
    return solution
