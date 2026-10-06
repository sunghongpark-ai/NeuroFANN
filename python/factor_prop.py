import numpy as np
from scipy import sparse
from scipy.linalg.lapack import dgecon, dgetrf

if __package__:
    from .shared import NeuroFANNError
else:
    from shared import NeuroFANNError


def factor_prop(operator):
    raw = operator.toarray() if sparse.issparse(operator) else np.asarray(operator)
    if (
        raw.ndim != 2
        or raw.shape[0] != raw.shape[1]
        or raw.shape[0] == 0
        or raw.dtype.kind not in "biuf"
        or not np.all(np.isfinite(raw))
    ):
        raise NeuroFANNError(
            "NeuroFANN:NonfinitePropagation", "The propagation matrix must be square with finite real values."
        )
    matrix = np.asarray(raw, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        matrix_norm = np.linalg.norm(matrix, ord=1)
    lu, pivot, info = dgetrf(matrix, overwrite_a=0)
    if info < 0 or not np.all(np.isfinite(lu)):
        raise NeuroFANNError(
            "NeuroFANN:NonfinitePropagation", "The propagation factorization produced nonfinite values."
        )
    reciprocal_condition, info = dgecon(lu, matrix_norm, norm="1")
    if info != 0:
        raise NeuroFANNError("NeuroFANN:NonfinitePropagation", "The propagation reciprocal condition estimate failed.")
    reciprocal_condition = float(reciprocal_condition)
    if not np.isfinite(reciprocal_condition) or reciprocal_condition <= np.finfo(np.float64).eps:
        raise NeuroFANNError(
            "NeuroFANN:SingularPropagation",
            "diag(Uprot) + Lppi is singular or numerically singular "
            f"(reciprocal condition {reciprocal_condition:.3g}).",
        )
    return {"LU": lu, "Pivot": pivot, "ReciprocalCondition": reciprocal_condition}
