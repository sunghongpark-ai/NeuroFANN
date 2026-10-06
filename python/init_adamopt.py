import numpy as np

if __package__:
    from .shared import NeuroFANNError, finite_scalar, integer_scalar
else:
    from shared import NeuroFANNError, finite_scalar, integer_scalar


def init_adamopt(num_var, alpha=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8):
    count = integer_scalar(num_var, "num_var", 1)
    identifier = "NeuroFANN:InvalidOptimizerState"
    alpha = finite_scalar(1e-4 if alpha is None else alpha, "alpha", identifier)
    beta1 = finite_scalar(0.9 if beta1 is None else beta1, "beta1", identifier)
    beta2 = finite_scalar(0.999 if beta2 is None else beta2, "beta2", identifier)
    epsilon = finite_scalar(1e-8 if epsilon is None else epsilon, "epsilon", identifier)
    if alpha <= 0 or epsilon <= 0:
        raise NeuroFANNError(identifier, "Adam alpha and epsilon must be positive.")
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise NeuroFANNError(identifier, "Adam beta1 and beta2 must lie in [0,1).")
    return {
        "alpha": alpha,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "t": 0,
        "m": np.zeros(count, dtype=np.float64),
        "v": np.zeros(count, dtype=np.float64),
    }
