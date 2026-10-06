from collections.abc import Mapping

import numpy as np

if __package__:
    from .shared import MAXIMUM_SEED, NeuroFANNError, finite_scalar, logical_flag
else:
    from shared import MAXIMUM_SEED, NeuroFANNError, finite_scalar, logical_flag


def model_options(overrides, dataset):
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise NeuroFANNError("NeuroFANN:InvalidParameter", "Parameter overrides must be a dictionary.")
    try:
        folds = np.asarray(dataset["CVindex"])
    except (KeyError, TypeError, ValueError) as exception:
        raise NeuroFANNError(
            "NeuroFANN:InvalidDataset", "Dataset must contain a normalized CVindex matrix."
        ) from exception
    if folds.ndim != 2 or folds.size == 0 or folds.dtype.kind not in "uif" or not np.all(np.isfinite(folds)):
        raise NeuroFANNError("NeuroFANN:InvalidDataset", "Dataset must contain a normalized CVindex matrix.")
    parameter = {
        "NumIter": folds.shape[0], "NumFold": float(np.max(folds[0])), "MaxEpoch": 500,
        "LearnRate": 0.001, "RegCoeff": 0.005, "Seed": 1,
        "StoreHistory": False, "Verbose": False,
    }
    for name, value in overrides.items():
        if name not in parameter:
            raise NeuroFANNError("NeuroFANN:InvalidParameter", f"Unknown parameter: {name}.")
        parameter[name] = value
    for name in ("NumIter", "NumFold", "MaxEpoch"):
        value = finite_scalar(parameter[name], name)
        if value < 1 or not value.is_integer():
            raise NeuroFANNError("NeuroFANN:InvalidParameter", f"{name} must be a positive integer scalar.")
        parameter[name] = int(value)
    if parameter["NumIter"] > folds.shape[0] or not 2 <= parameter["NumFold"] <= folds.shape[1]:
        raise NeuroFANNError(
            "NeuroFANN:InvalidParameter", "NumIter exceeds CVindex or NumFold is outside 2:NumSamples."
        )
    expected = np.arange(1, parameter["NumFold"] + 1)
    for iteration in range(parameter["NumIter"]):
        if not np.array_equal(np.unique(folds[iteration]), expected):
            raise NeuroFANNError(
                "NeuroFANN:InvalidFold", f"CVindex row {iteration + 1} must include exactly the fold IDs 1:NumFold."
            )
    for name in ("LearnRate", "RegCoeff", "Seed"):
        parameter[name] = finite_scalar(parameter[name], name)
    if parameter["LearnRate"] <= 0 or parameter["RegCoeff"] < 0:
        raise NeuroFANNError("NeuroFANN:InvalidParameter", "LearnRate must be positive and RegCoeff nonnegative.")
    seed = parameter["Seed"]
    if seed < 0 or not seed.is_integer() or seed + parameter["NumIter"] - 1 > MAXIMUM_SEED:
        raise NeuroFANNError(
            "NeuroFANN:InvalidParameter", "Seed through Seed+NumIter-1 must be integers in the uint32 range."
        )
    parameter["Seed"] = int(seed)
    for name in ("StoreHistory", "Verbose"):
        parameter[name] = logical_flag(parameter[name], name)
    return parameter
