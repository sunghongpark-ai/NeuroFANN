import numpy as np

if __package__:
    from .init_adamopt import _finite_scalar, _positive_integer
    from .shared import NeuroFANNError
else:
    from init_adamopt import _finite_scalar, _positive_integer
    from shared import NeuroFANNError


def _history_flag(value):
    if value is None:
        return False
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "biuf" or array.item() not in (0, 1):
        raise NeuroFANNError("NeuroFANN:InvalidParameter", "StoreHistory must be a scalar logical flag.")
    return bool(array)


def param_initialize(model):
    initialized = dict(model)
    proteins = _positive_integer(model["NumProtein"], "NumProtein")
    clusters = _positive_integer(model["NumCluster"], "NumCluster")
    epochs = _positive_integer(model["MaxEpoch"], "MaxEpoch")
    seed_value = model.get("Seed")
    if seed_value is None:
        seed_value = model["IdxIter"]
    seed = _finite_scalar(seed_value, "Seed", "NeuroFANN:InvalidParameter")
    if seed < 0 or seed > np.iinfo(np.uint32).max or not seed.is_integer():
        raise NeuroFANNError("NeuroFANN:InvalidParameter", "Seed must be an integer in the uint32 range.")
    seed = int(seed)
    store_history = _history_flag(model.get("StoreHistory", False))
    stream = np.random.RandomState(5489 if seed == 0 else seed)
    scale = np.sqrt(6.0 / (clusters + 1))
    initialized.update({
        "NumProtein": proteins,
        "NumCluster": clusters,
        "MaxEpoch": epochs,
        "Seed": seed,
        "StoreHistory": store_history,
        "Uprot": np.ones(proteins, dtype=np.float64),
        "Aclus": np.zeros(proteins, dtype=np.float64),
        "Babt": (2 * stream.rand(clusters) - 1) * scale,
        "Bmta": (2 * stream.rand(clusters) - 1) * scale,
        "Bwmh": (2 * stream.rand(clusters) - 1) * scale,
        "SizeParam": np.array([[proteins, 1], [proteins, 1], [clusters, 1], [clusters, 1], [clusters, 1]], dtype=np.int64),
        "NumParam": 2 * proteins + 3 * clusters,
        "WeightEpoch": [None] * epochs if store_history else [],
        "LossTrain": np.full(epochs, np.nan, dtype=np.float64),
        "LossValid": np.full(epochs, np.nan, dtype=np.float64),
        "ObjectiveTrain": np.full(epochs, np.nan, dtype=np.float64),
    })
    initialized["WeightParam"] = np.concatenate([initialized[name] for name in ("Uprot", "Aclus", "Babt", "Bmta", "Bwmh")])
    return initialized
