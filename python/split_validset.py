from copy import deepcopy

import numpy as np
from scipy.sparse import issparse

if __package__:
    from .shared import NeuroFANNError
else:
    from shared import NeuroFANNError


def _invalid(message):
    raise NeuroFANNError("NeuroFANN:InvalidDataset", message)


def _numeric(value, name, allow_bool=False):
    try:
        array = np.asarray(value.toarray() if issparse(value) else value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NeuroFANNError("NeuroFANN:InvalidDataset", f"{name} must be real numeric data.") from exc
    allowed = "buif" if allow_bool else "uif"
    if array.dtype.kind not in allowed:
        _invalid(f"{name} must be real numeric data.")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(array, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        _invalid(f"{name} must contain finite values.")
    return result


def _count(value, name):
    array = _numeric(value, name)
    if array.size != 1:
        _invalid(f"{name} must be a positive integer scalar.")
    scalar = float(array.reshape(-1)[0])
    if scalar < 1 or scalar != np.floor(scalar):
        _invalid(f"{name} must be a positive integer scalar.")
    return int(scalar)


def _matrix(value, name):
    array = _numeric(value, name)
    if array.ndim != 2:
        _invalid(f"{name} must be a two-dimensional matrix.")
    return array


def _is_vector(array):
    return array.ndim <= 1 or (array.ndim == 2 and 1 in array.shape)


def _empty(value):
    if value is None:
        return True
    if issparse(value):
        return 0 in value.shape
    try:
        return np.asarray(value).size == 0
    except (TypeError, ValueError):
        return False


def _labels(value, count, name):
    array = _numeric(value, name, allow_bool=True)
    if not _is_vector(array) or array.size != count or np.any((array != 0) & (array != 1)):
        _invalid(f"{name} must contain one binary label per sample.")
    return array.reshape(-1).copy()


def split_validset(dataset):
    required = (
        "XData", "XTest", "YabtData", "YmtaData", "YwmhData", "Lppi",
        "NumProtein", "IdxCluster", "NumCluster", "CVindex",
    )
    if not isinstance(dataset, dict):
        _invalid("Dataset must be a dictionary.")
    missing = [name for name in required if name not in dataset]
    if missing:
        _invalid("Dataset is missing: " + ", ".join(missing) + ".")
    proteins = _count(dataset["NumProtein"], "NumProtein")
    clusters = _count(dataset["NumCluster"], "NumCluster")
    if clusters > proteins:
        _invalid("NumCluster cannot exceed NumProtein.")
    data = _matrix(dataset["XData"], "XData")
    test = _matrix(dataset["XTest"], "XTest")
    if data.shape[0] != proteins or test.shape[0] != proteins or data.shape[1] < 2:
        _invalid("XData and XTest require NumProtein rows; XData needs at least two samples.")
    laplacian = _matrix(dataset["Lppi"], "Lppi")
    if laplacian.shape != (proteins, proteins):
        _invalid("Lppi must be NumProtein-by-NumProtein.")
    cluster_index = _numeric(dataset["IdxCluster"], "IdxCluster")
    if (
        not _is_vector(cluster_index)
        or cluster_index.size != proteins
        or np.any(cluster_index != np.floor(cluster_index))
        or np.any((cluster_index < 1) | (cluster_index > clusters))
    ):
        _invalid("IdxCluster must assign each protein an integer cluster in 1:NumCluster.")
    cluster_index = cluster_index.reshape(-1).astype(np.int64)
    if np.any(np.bincount(cluster_index, minlength=clusters + 1)[1:] == 0):
        _invalid("Every cluster must contain at least one protein.")
    result = deepcopy(dataset)
    result.update(
        NumProtein=proteins, NumCluster=clusters, XData=data, XTest=test,
        Lppi=laplacian, IdxCluster=cluster_index,
    )
    for target in ("abt", "mta", "wmh"):
        data_name = "Y" + target + "Data"
        test_name = "Y" + target + "Test"
        result[data_name] = _labels(dataset[data_name], data.shape[1], data_name)
        test_labels = dataset.get(test_name)
        result[test_name] = np.empty(0, dtype=np.float64) if _empty(test_labels) else _labels(test_labels, test.shape[1], test_name)
    folds = _numeric(dataset["CVindex"], "CVindex")
    if folds.ndim == 1:
        folds = folds.reshape(1, -1)
    if (
        folds.ndim != 2 or folds.size == 0 or folds.shape[1] != data.shape[1]
        or np.any(folds < 1) or np.any(folds != np.floor(folds))
    ):
        _invalid("CVindex must contain positive integer fold IDs with one column per XData sample.")
    result["CVindex"] = folds.copy()
    adjacency = dataset.get("Wppi")
    if _empty(adjacency):
        result["Wppi"] = np.empty((0, 0), dtype=np.float64)
    else:
        adjacency = _matrix(adjacency, "Wppi")
        if adjacency.shape != (proteins, proteins):
            _invalid("Wppi must be NumProtein-by-NumProtein when supplied.")
        result["Wppi"] = adjacency
    identifiers = dataset.get("IdxProtein")
    if _empty(identifiers):
        result["IdxProtein"] = np.array([str(index + 1) for index in range(proteins)], dtype=object)
    else:
        try:
            identifiers = np.asarray(identifiers)
        except (TypeError, ValueError) as exc:
            raise NeuroFANNError("NeuroFANN:InvalidDataset", "IdxProtein must contain one identifier per protein.") from exc
        if not _is_vector(identifiers) or identifiers.size != proteins:
            _invalid("IdxProtein must contain one identifier per protein.")
        result["IdxProtein"] = deepcopy(identifiers.reshape(-1))
    return result