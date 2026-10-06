import numpy as np

if __package__:
    from .rand_mt19937 import rand_mt19937
    from .shared import MAXIMUM_SEED, integer_scalar, logical_flag
else:
    from rand_mt19937 import rand_mt19937
    from shared import MAXIMUM_SEED, integer_scalar, logical_flag


def param_initialize(model):
    initialized = dict(model)
    proteins = integer_scalar(model["NumProtein"], "NumProtein", 1)
    clusters = integer_scalar(model["NumCluster"], "NumCluster", 1)
    epochs = integer_scalar(model["MaxEpoch"], "MaxEpoch", 1)
    seed_value = model.get("Seed")
    if seed_value is None:
        seed_value = model["IdxIter"]
    seed = integer_scalar(seed_value, "Seed", 0, MAXIMUM_SEED)
    history_value = model.get("StoreHistory")
    store_history = False if history_value is None else logical_flag(history_value, "StoreHistory")
    draws = rand_mt19937(seed, 3 * clusters)
    scale = np.sqrt(6.0 / (clusters + 1))
    initialized.update({
        "NumProtein": proteins,
        "NumCluster": clusters,
        "MaxEpoch": epochs,
        "Seed": seed,
        "StoreHistory": store_history,
        "Uprot": np.ones(proteins, dtype=np.float64),
        "Aclus": np.zeros(proteins, dtype=np.float64),
        "Babt": (2 * draws[:clusters] - 1) * scale,
        "Bmta": (2 * draws[clusters:2 * clusters] - 1) * scale,
        "Bwmh": (2 * draws[2 * clusters:] - 1) * scale,
        "SizeParam": np.array(
            [[proteins, 1], [proteins, 1], [clusters, 1], [clusters, 1], [clusters, 1]], dtype=np.int64
        ),
        "NumParam": 2 * proteins + 3 * clusters,
        "WeightEpoch": [None] * epochs if store_history else [],
        "LossTrain": np.full(epochs, np.nan, dtype=np.float64),
        "LossValid": np.full(epochs, np.nan, dtype=np.float64),
        "ObjectiveTrain": np.full(epochs, np.nan, dtype=np.float64),
    })
    initialized["WeightParam"] = np.concatenate(
        [initialized[name] for name in ("Uprot", "Aclus", "Babt", "Bmta", "Bwmh")]
    )
    return initialized
