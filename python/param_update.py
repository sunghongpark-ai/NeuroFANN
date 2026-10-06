from collections.abc import Mapping

import numpy as np

if __package__:
    from .shared import NeuroFANNError, finite_scalar
else:
    from shared import NeuroFANNError, finite_scalar

OPTIMIZER_FIELDS = ("alpha", "beta1", "beta2", "epsilon", "t", "m", "v")


def floating_vector(value, name, identifier, shape=None):
    array = np.asarray(value)
    if array.dtype.kind != "f" or array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise NeuroFANNError(identifier, f"{name} must be a nonempty finite real floating-point vector.")
    if shape is not None and array.shape != shape:
        raise NeuroFANNError(identifier, f"{name} must have the same shape as WeightParam.")
    return np.asarray(array, dtype=np.float64)


def validate_optimizer(optimizer, shape):
    identifier = "NeuroFANN:InvalidOptimizerState"
    if not isinstance(optimizer, Mapping) or not set(OPTIMIZER_FIELDS).issubset(optimizer):
        raise NeuroFANNError(identifier, "AdamParam must contain all fields created by init_adamopt.")
    parameter = dict(optimizer)
    for name in ("alpha", "beta1", "beta2", "epsilon", "t"):
        parameter[name] = finite_scalar(parameter[name], f"AdamParam.{name}", identifier)
    if parameter["alpha"] <= 0 or parameter["epsilon"] <= 0:
        raise NeuroFANNError(identifier, "Adam alpha and epsilon must be positive.")
    if not 0 <= parameter["beta1"] < 1 or not 0 <= parameter["beta2"] < 1:
        raise NeuroFANNError(identifier, "Adam beta1 and beta2 must lie in [0,1).")
    step = parameter["t"]
    if step < 0 or step >= 2**53 or not step.is_integer():
        raise NeuroFANNError(identifier, "Adam t must be a nonnegative integer less than 2**53.")
    parameter["t"] = int(step)
    parameter["m"] = floating_vector(parameter["m"], "AdamParam.m", identifier, shape)
    parameter["v"] = floating_vector(parameter["v"], "AdamParam.v", identifier, shape)
    if np.any(parameter["v"] < 0):
        raise NeuroFANNError(identifier, "Adam v must be nonnegative.")
    return parameter


def copy_optimizer(optimizer):
    return {name: value.copy() if isinstance(value, np.ndarray) else value for name, value in optimizer.items()}


def param_update(model):
    weights = floating_vector(model["WeightParam"], "WeightParam", "NeuroFANN:InvalidWeights")
    gradient = floating_vector(model["Gradient"], "Gradient", "NeuroFANN:InvalidGradient", weights.shape)
    parameter = validate_optimizer(model.get("AdamParam"), weights.shape)
    parameter["t"] += 1
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        parameter["m"] = parameter["beta1"] * parameter["m"] + (1 - parameter["beta1"]) * gradient
        parameter["v"] = parameter["beta2"] * parameter["v"] + (1 - parameter["beta2"]) * gradient**2
        parameter["m_hat"] = parameter["m"] / (1 - parameter["beta1"] ** parameter["t"])
        parameter["v_hat"] = parameter["v"] / (1 - parameter["beta2"] ** parameter["t"])
        updated_weights = weights - parameter["alpha"] * parameter["m_hat"] / (
            np.sqrt(parameter["v_hat"]) + parameter["epsilon"]
        )
    if any(not np.all(np.isfinite(parameter[name])) for name in ("m", "v", "m_hat", "v_hat")) or not np.all(
        np.isfinite(updated_weights)
    ):
        raise NeuroFANNError(
            "NeuroFANN:NonfiniteOptimizerUpdate",
            "Adam produced nonfinite state or weights. Check the gradient magnitude, learning rate, and input scaling.",
        )
    updated = dict(model)
    updated["AdamParam"] = parameter
    updated["WeightParam"] = updated_weights
    return updated
