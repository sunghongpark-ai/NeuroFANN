from collections.abc import Mapping

import numpy as np
from scipy import sparse

if __package__:
    from .model_options import model_options
    from .shared import NeuroFANNError, TARGETS
    from .split_validset import split_validset
else:
    from model_options import model_options
    from shared import NeuroFANNError, TARGETS
    from split_validset import split_validset


def _index(value, maximum, identifier, message):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exception:
        raise NeuroFANNError(identifier, message) from exception
    if array.dtype.kind not in "uif" or array.size != 1:
        raise NeuroFANNError(identifier, message)
    scalar = float(array.reshape(-1)[0])
    if not np.isfinite(scalar) or scalar != np.floor(scalar) or scalar < 1 or scalar > maximum:
        raise NeuroFANNError(identifier, message)
    return int(scalar)


def model_initialize(model, fold):
    if not isinstance(model, Mapping) or not all(name in model for name in ("Dataset", "Parameter", "IdxIter")):
        raise NeuroFANNError("NeuroFANN:InvalidDataset", "Model must contain Dataset, Parameter, and IdxIter.")
    dataset = split_validset(model["Dataset"])
    parameter = model_options(model["Parameter"], dataset)
    iteration = _index(
        model["IdxIter"], parameter["NumIter"], "NeuroFANN:InvalidParameter",
        "IdxIter must identify an available cross-validation iteration.",
    )
    fold = _index(fold, parameter["NumFold"], "NeuroFANN:InvalidFold", "Fold must be an integer between 1 and NumFold.")
    fold_ids = dataset["CVindex"][iteration - 1]
    train = np.flatnonzero(fold_ids != fold)
    valid = np.flatnonzero(fold_ids == fold)
    result = {
        "IdxIter": iteration, "IdxFold": fold, "NumFold": parameter["NumFold"],
        "IdxTrain": train, "IdxValid": valid, "NumTrain": int(train.size),
        "NumValid": int(valid.size), "NumTest": int(dataset["XTest"].shape[1]),
        "XTrain": dataset["XData"][:, train].copy(), "XValid": dataset["XData"][:, valid].copy(),
        "XTest": dataset["XTest"].copy(),
    }
    for target in TARGETS:
        prefix = "Y" + target
        result[prefix + "Train"] = dataset[prefix + "Data"][train].copy()
        result[prefix + "Valid"] = dataset[prefix + "Data"][valid].copy()
        result[prefix + "Test"] = dataset[prefix + "Test"].copy()
    for name in ("Wppi", "Lppi", "IdxProtein", "NumProtein", "IdxCluster", "NumCluster"):
        result[name] = dataset[name]
    proteins = dataset["NumProtein"]
    clusters = dataset["NumCluster"]
    result["ClusterMembers"] = [np.flatnonzero(dataset["IdxCluster"] == cluster) for cluster in range(1, clusters + 1)]
    result["ClusterAssignment"] = sparse.csr_matrix(
        (np.ones(proteins, dtype=np.float64), (dataset["IdxCluster"] - 1, np.arange(proteins))),
        shape=(clusters, proteins),
    )
    for name in ("MaxEpoch", "LearnRate", "RegCoeff", "StoreHistory"):
        result[name] = parameter[name]
    result["Seed"] = parameter["Seed"] + iteration - 1
    return result
