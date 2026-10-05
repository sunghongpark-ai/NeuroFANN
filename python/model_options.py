import numpy as np

if __package__:
    from .shared import NeuroFANNError
else:
    from shared import NeuroFANNError


def _scalar(value, name, allow_bool=False):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NeuroFANNError("NeuroFANN:InvalidParameter", f"{name} must be a finite real scalar.") from exc
    allowed = "buif" if allow_bool else "uif"
    if array.dtype.kind not in allowed or array.size != 1:
        raise NeuroFANNError("NeuroFANN:InvalidParameter", f"{name} must be a finite real scalar.")
    scalar = float(array.reshape(-1)[0])
    if not np.isfinite(scalar):
        raise NeuroFANNError("NeuroFANN:InvalidParameter", f"{name} must be a finite real scalar.")
    return scalar


def model_options(overrides, dataset):
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise NeuroFANNError("NeuroFANN:InvalidParameter", "Parameter overrides must be a dictionary.")
    try:
        folds = np.asarray(dataset["CVindex"])
    except (KeyError, TypeError, ValueError) as exc:
        raise NeuroFANNError("NeuroFANN:InvalidDataset", "Dataset must contain a normalized CVindex matrix.") from exc
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
        value = _scalar(parameter[name], name)
        if value < 1 or value != np.floor(value):
            raise NeuroFANNError("NeuroFANN:InvalidParameter", f"{name} must be a positive integer scalar.")
        parameter[name] = int(value)
    if parameter["NumIter"] > folds.shape[0] or not 2 <= parameter["NumFold"] <= folds.shape[1]:
        raise NeuroFANNError("NeuroFANN:InvalidParameter", "NumIter exceeds CVindex or NumFold is outside 2:NumSamples.")
    expected = np.arange(1, parameter["NumFold"] + 1)
    for iteration in range(parameter["NumIter"]):
        if not np.array_equal(np.unique(folds[iteration]), expected):
            raise NeuroFANNError("NeuroFANN:InvalidFold", f"CVindex row {iteration + 1} must include exactly the fold IDs 1:NumFold.")
    for name in ("LearnRate", "RegCoeff", "Seed"):
        parameter[name] = _scalar(parameter[name], name)
    if parameter["LearnRate"] <= 0 or parameter["RegCoeff"] < 0:
        raise NeuroFANNError("NeuroFANN:InvalidParameter", "LearnRate must be positive and RegCoeff nonnegative.")
    seed = parameter["Seed"]
    if seed < 0 or seed != np.floor(seed) or seed + parameter["NumIter"] - 1 > 4294967295:
        raise NeuroFANNError("NeuroFANN:InvalidParameter", "Seed through Seed+NumIter-1 must be integers in the uint32 range.")
    parameter["Seed"] = int(seed)
    for name in ("StoreHistory", "Verbose"):
        value = _scalar(parameter[name], name, allow_bool=True)
        if value not in (0, 1):
            raise NeuroFANNError("NeuroFANN:InvalidParameter", f"{name} must be a scalar logical flag.")
        parameter[name] = bool(value)
    return parameter