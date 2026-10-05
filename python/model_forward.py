import numpy as np
from scipy import sparse

if __package__:
    from .shared import NeuroFANNError
    from .solver_prop import factor_prop, solver_prop
else:
    from shared import NeuroFANNError
    from solver_prop import factor_prop, solver_prop


def model_forward(model, include_test=True):
    flag = np.asarray(include_test)
    if flag.ndim != 0 or flag.dtype.kind not in "biuf" or flag not in (0, 1):
        raise NeuroFANNError(
            "NeuroFANN:InvalidParameter", "include_test must be a scalar logical value."
        )
    result = dict(model)
    protein_count = model["NumProtein"]
    cluster_count = model["NumCluster"]
    laplacian = model["Lppi"]
    if sparse.issparse(laplacian):
        laplacian = laplacian.toarray()
    with np.errstate(over="ignore", invalid="ignore"):
        operator = model["Udiag"] + np.asarray(laplacian)
    factor, reciprocal_condition = factor_prop(operator)
    result["PropagationFactor"] = factor
    result["PropagationReciprocalCondition"] = reciprocal_condition

    assignment = model.get("ClusterAssignment")
    if assignment is None or assignment.shape != (cluster_count, protein_count):
        assignment = sparse.csr_matrix(
            (
                np.ones(protein_count, dtype=np.float64),
                (
                    np.asarray(model["IdxCluster"], dtype=np.int64).reshape(-1) - 1,
                    np.arange(protein_count),
                ),
            ),
            shape=(cluster_count, protein_count),
        )
        result["ClusterAssignment"] = assignment
    if sparse.issparse(assignment):
        pooling = assignment.multiply(model["Aprob"][None, :]).tocsr()
    else:
        pooling = np.asarray(assignment) * model["Aprob"][None, :]
    heads = np.column_stack((model["Babt"], model["Bmta"], model["Bwmh"]))
    splits = ("Train", "Valid", "Test") if include_test else ("Train", "Valid")
    if not include_test:
        for name in ("HTest", "ZTest", "LogitsTest", "PabtTest", "PmtaTest", "PwmhTest"):
            result.pop(name, None)

    features_by_split = {}
    for split in splits:
        raw_features = np.asarray(model[f"X{split}"])
        if (
            raw_features.ndim != 2
            or raw_features.shape[0] != protein_count
            or raw_features.dtype.kind not in "biuf"
            or not np.isfinite(raw_features).all()
        ):
            raise NeuroFANNError(
                "NeuroFANN:InvalidFeatures",
                "Each feature matrix must contain finite real values with one row per protein.",
            )
        features_by_split[split] = np.asarray(raw_features, dtype=np.float64)
    sample_count = sum(features.shape[1] for features in features_by_split.values())
    use_transform = sample_count >= protein_count
    propagation = solver_prop(factor, model["Udiag"]) if use_transform else None

    for split, features in features_by_split.items():
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            hidden = (
                propagation @ features
                if use_transform
                else solver_prop(factor, model["Uprot"][:, None] * features)
            )
            if not np.isfinite(hidden).all():
                raise NeuroFANNError(
                    "NeuroFANN:NonfinitePropagation",
                    "Propagation produced nonfinite hidden features.",
                )
            pooled = pooling @ hidden
            logits = heads.T @ pooled
        if not np.isfinite(logits).all():
            raise NeuroFANNError(
                "NeuroFANN:NonfinitePropagation",
                "Forward propagation produced nonfinite logits.",
            )
        probabilities = np.empty_like(logits)
        positive = logits >= 0
        with np.errstate(under="ignore"):
            probabilities[positive] = 1 / (1 + np.exp(-logits[positive]))
            negative_exp = np.exp(logits[~positive])
        probabilities[~positive] = negative_exp / (1 + negative_exp)
        result[f"H{split}"] = hidden
        result[f"Z{split}"] = pooled
        result[f"Logits{split}"] = logits
        for index, name in enumerate(("Pabt", "Pmta", "Pwmh")):
            result[f"{name}{split}"] = probabilities[index].copy()
    return result
