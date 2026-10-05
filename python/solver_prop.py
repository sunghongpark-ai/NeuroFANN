import warnings

import numpy as np
from scipy import sparse
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve
from scipy.linalg.lapack import get_lapack_funcs

if __package__:
    from .shared import NeuroFANNError
else:
    from shared import NeuroFANNError


def factor_prop(matrix):
    raw_matrix = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    if (
        raw_matrix.ndim != 2
        or raw_matrix.shape[0] != raw_matrix.shape[1]
        or raw_matrix.shape[0] == 0
        or raw_matrix.dtype.kind not in "biuf"
        or not np.isfinite(raw_matrix).all()
    ):
        raise NeuroFANNError(
            "NeuroFANN:NonfinitePropagation",
            "The propagation matrix must be square with finite real values.",
        )
    operator = np.asarray(raw_matrix, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        matrix_norm = np.linalg.norm(operator, ord=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LinAlgWarning)
        factor = lu_factor(operator, overwrite_a=False, check_finite=False)
    if not np.isfinite(factor[0]).all():
        raise NeuroFANNError(
            "NeuroFANN:NonfinitePropagation",
            "The propagation factorization produced nonfinite values.",
        )
    estimate_condition = get_lapack_funcs("gecon", (factor[0],))
    reciprocal_condition, info = estimate_condition(factor[0], matrix_norm, norm="1")
    if info != 0:
        raise NeuroFANNError(
            "NeuroFANN:NonfinitePropagation",
            "The propagation reciprocal condition estimate failed.",
        )
    if (
        not np.isfinite(reciprocal_condition)
        or reciprocal_condition <= np.finfo(np.float64).eps
    ):
        raise NeuroFANNError(
            "NeuroFANN:SingularPropagation",
            "diag(Uprot) + Lppi is singular or numerically singular "
            f"(reciprocal condition {reciprocal_condition:.3g}).",
        )
    return factor, float(reciprocal_condition)


def solver_prop(factor, right_hand_side, transpose_system=False):
    raw_rhs = (
        right_hand_side.toarray()
        if sparse.issparse(right_hand_side)
        else np.asarray(right_hand_side)
    )
    if (
        raw_rhs.ndim not in (1, 2)
        or raw_rhs.shape[0] != factor[0].shape[0]
        or raw_rhs.dtype.kind not in "biuf"
        or not np.isfinite(raw_rhs).all()
    ):
        raise NeuroFANNError(
            "NeuroFANN:NonfinitePropagation",
            "The propagation right-hand side must have a matching shape and finite real values.",
        )
    rhs = np.asarray(raw_rhs, dtype=np.float64)
    solution = lu_solve(
        factor,
        rhs,
        trans=1 if transpose_system else 0,
        overwrite_b=False,
        check_finite=False,
    )
    if not np.isfinite(solution).all():
        raise NeuroFANNError(
            "NeuroFANN:NonfinitePropagation",
            "The propagation solve produced nonfinite values.",
        )
    return solution
