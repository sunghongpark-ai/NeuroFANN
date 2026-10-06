from collections.abc import Mapping
from copy import deepcopy

import numpy as np
from scipy.sparse import issparse

if __package__:
    from .ppi_laplacian import ppi_laplacian
    from .shared import NeuroFANNError, TARGETS
else:
    from ppi_laplacian import ppi_laplacian
    from shared import NeuroFANNError, TARGETS


def _invalid(message):
    raise NeuroFANNError("NeuroFANN:InvalidDataset", message)


def _numeric(value, name, allow_bool=False):
    try:
        array = np.asarray(value.toarray() if issparse(value) else value)
    except (TypeError, ValueError, OverflowError) as exception:
        raise NeuroFANNError("NeuroFANN:InvalidDataset", f"{name} must be real numeric data.") from exception
    allowed = "buif" if allow_bool else "uif"
    if array.dtype.kind not in allowed:
        _invalid(f"{name} must be real numeric data.")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.array(array, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(converted)):
        _invalid(f"{name} must contain finite values.")
    return converted


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
    if isinstance(value, np.ndarray):
        return value.size == 0
    try:
        return np.asarray(value, dtype=object).size == 0
    except (TypeError, ValueError):
        return False


def _labels(value, count, name):
    array = _numeric(value, name, allow_bool=True)
    if not _is_vector(array) or array.size != count or np.any((array != 0) & (array != 1)):
        _invalid(f"{name} must contain one binary label per sample.")
    return array.reshape(-1).copy()


def _text_column(value, count, name):
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exception:
        raise NeuroFANNError(
            "NeuroFANN:InvalidDataset", f"{name} must contain text or numeric identifiers."
        ) from exception
    if not _is_vector(array) or array.size != count:
        _invalid(f"{name} must contain one identifier per entry.")
    names = []
    kinds = set()
    for item in array.reshape(-1):
        if isinstance(item, (str, np.str_)):
            kinds.add("text")
            names.append(str(item))
        elif isinstance(item, (bool, int, float, np.bool_, np.integer, np.floating)) and np.isfinite(float(item)):
            kinds.add("number")
            names.append(format(float(item), ".17g"))
        else:
            _invalid(f"{name} must contain text or finite numeric identifiers.")
    if len(kinds) > 1:
        _invalid(f"{name} must contain text or finite numeric identifiers.")
    return np.array(names, dtype=object)


def _default_names(prefix, count, minimum_width):
    if minimum_width > 0:
        width = max(minimum_width, len(str(count)))
        return np.array([f"{prefix}{index:0{width}d}" for index in range(1, count + 1)], dtype=object)
    return np.array([f"{prefix}{index}" for index in range(1, count + 1)], dtype=object)


def split_validset(dataset, require_cv=True):
    if not isinstance(dataset, Mapping):
        _invalid("Dataset must be a dictionary.")
    required = ("XData", "XTest", "YabtData", "YmtaData", "YwmhData", "NumProtein", "IdxCluster", "NumCluster")
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
    sample_count = data.shape[1]
    test_count = test.shape[1]
    has_adjacency = not _empty(dataset.get("Wppi"))
    has_laplacian = not _empty(dataset.get("Lppi"))
    if has_adjacency:
        adjacency = _matrix(dataset["Wppi"], "Wppi")
        if adjacency.shape != (proteins, proteins):
            _invalid("Wppi must be NumProtein-by-NumProtein when supplied.")
    else:
        adjacency = np.empty((0, 0), dtype=np.float64)
    if has_laplacian:
        laplacian = _matrix(dataset["Lppi"], "Lppi")
        if laplacian.shape != (proteins, proteins):
            _invalid("Lppi must be NumProtein-by-NumProtein.")
    elif has_adjacency:
        laplacian = ppi_laplacian(adjacency)
    else:
        _invalid("Dataset requires Lppi or the PPI weights Wppi.")
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
    normalized = {
        "NumProtein": proteins, "NumCluster": clusters, "XData": data, "XTest": test,
        "Wppi": adjacency, "Lppi": laplacian, "IdxCluster": cluster_index,
    }
    for target in TARGETS:
        data_name = "Y" + target + "Data"
        test_name = "Y" + target + "Test"
        normalized[data_name] = _labels(dataset[data_name], sample_count, data_name)
        test_labels = dataset.get(test_name)
        normalized[test_name] = (
            np.empty(0, dtype=np.float64) if _empty(test_labels) else _labels(test_labels, test_count, test_name)
        )
    if _empty(dataset.get("CVindex")):
        if require_cv:
            _invalid("Dataset has no cross-validation folds (CVindex); create them with split_cvindex.")
        normalized["CVindex"] = np.empty((0, sample_count), dtype=np.float64)
    else:
        folds = _numeric(dataset["CVindex"], "CVindex")
        if folds.ndim == 1:
            folds = folds.reshape(1, -1)
        if (
            folds.ndim != 2 or folds.shape[1] != sample_count
            or np.any(folds < 1) or np.any(folds != np.floor(folds))
        ):
            _invalid("CVindex must contain positive integer fold IDs with one column per XData sample.")
        normalized["CVindex"] = folds
    if _empty(dataset.get("IdxProtein")):
        normalized["IdxProtein"] = _default_names("", proteins, 0)
    else:
        normalized["IdxProtein"] = _text_column(dataset["IdxProtein"], proteins, "IdxProtein")
    if _empty(dataset.get("IdxSampleData")):
        normalized["IdxSampleData"] = _default_names("D", sample_count, 3)
    else:
        normalized["IdxSampleData"] = _text_column(dataset["IdxSampleData"], sample_count, "IdxSampleData")
    if _empty(dataset.get("IdxSampleTest")):
        normalized["IdxSampleTest"] = _default_names("V", test_count, 3)
    else:
        normalized["IdxSampleTest"] = _text_column(dataset["IdxSampleTest"], test_count, "IdxSampleTest")
    if len(set(normalized["IdxProtein"])) != proteins:
        _invalid("IdxProtein identifiers must be unique.")
    if len(set(normalized["IdxSampleData"]) | set(normalized["IdxSampleTest"])) != sample_count + test_count:
        _invalid("Sample identifiers must be unique across XData and XTest.")
    result = {name: deepcopy(value) for name, value in dataset.items() if name not in normalized}
    result.update(normalized)
    return result
