import numpy as np

if __package__:
    from .shared import NeuroFANNError
else:
    from shared import NeuroFANNError


def _finite_scalar(value, name, identifier="NeuroFANN:InvalidOptimizerState"):
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iuf":
        raise NeuroFANNError(identifier, f"{name} must be a finite real numeric scalar.")
    number = float(array)
    if not np.isfinite(number):
        raise NeuroFANNError(identifier, f"{name} must be a finite real numeric scalar.")
    return number


def _positive_integer(value, name):
    number = _finite_scalar(value, name, "NeuroFANN:InvalidParameter")
    if number < 1 or not number.is_integer():
        raise NeuroFANNError("NeuroFANN:InvalidParameter", f"{name} must be a positive integer scalar.")
    return int(number)


def init_adamopt(num_var, alpha=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8):
    count = _positive_integer(num_var, "num_var")
    alpha = _finite_scalar(1e-4 if alpha is None else alpha, "alpha")
    beta1 = _finite_scalar(0.9 if beta1 is None else beta1, "beta1")
    beta2 = _finite_scalar(0.999 if beta2 is None else beta2, "beta2")
    epsilon = _finite_scalar(1e-8 if epsilon is None else epsilon, "epsilon")
    if alpha <= 0 or epsilon <= 0:
        raise NeuroFANNError("NeuroFANN:InvalidOptimizerState", "Adam alpha and epsilon must be positive.")
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise NeuroFANNError("NeuroFANN:InvalidOptimizerState", "Adam beta1 and beta2 must lie in [0,1).")
    return {
        "alpha": alpha,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "t": 0,
        "m": np.zeros(count, dtype=np.float64),
        "v": np.zeros(count, dtype=np.float64),
    }
