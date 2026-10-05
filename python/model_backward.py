import numpy as np

if __package__:
    from .shared import NeuroFANNError
    from .solver_prop import solver_prop
else:
    from shared import NeuroFANNError
    from solver_prop import solver_prop


def model_backward(model):
    if "PropagationFactor" not in model:
        raise NeuroFANNError(
            "NeuroFANN:MissingForwardPass",
            "Run model_forward after the latest parameter reshape before model_backward.",
        )
    result = dict(model)
    count = model["NumTrain"]
    logits = np.asarray(model["LogitsTrain"])
    labels = [
        np.asarray(model[f"{name}Train"]).reshape(-1)
        for name in ("Yabt", "Ymta", "Ywmh")
    ]
    if (
        count < 1
        or logits.shape != (3, count)
        or logits.dtype.kind not in "biuf"
        or not np.isfinite(logits).all()
        or any(
            label.size != count
            or label.dtype.kind not in "biuf"
            or not np.isfinite(label).all()
            or np.any((label < 0) | (label > 1))
            for label in labels
        )
    ):
        raise NeuroFANNError(
            "NeuroFANN:InvalidTargets",
            "Training labels must match finite logits and lie in the interval [0, 1].",
        )
    targets = np.vstack(labels).astype(np.float64, copy=False)
    residual = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    with np.errstate(under="ignore"):
        negative_exp = np.exp(-logits[positive])
        residual[positive] = (1 - targets[positive]) - negative_exp / (1 + negative_exp)
        positive_exp = np.exp(logits[~positive])
        residual[~positive] = positive_exp / (1 + positive_exp) - targets[~positive]
    residual /= count

    heads = np.column_stack((model["Babt"], model["Bmta"], model["Bwmh"]))
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        head_gradient = model["ZTrain"] @ residual.T
        pooled_gradient = heads @ residual
        attention_gradient = np.zeros_like(model["Aclus"], dtype=np.float64)
        hidden_gradient = np.zeros_like(model["HTrain"], dtype=np.float64)
        for cluster, members in enumerate(model["ClusterMembers"]):
            probabilities = model["Aprob"][members]
            probability_gradient = model["HTrain"][members] @ pooled_gradient[cluster]
            attention_gradient[members] = probabilities * (
                probability_gradient - probabilities @ probability_gradient
            )
            hidden_gradient[members] = probabilities[:, None] * pooled_gradient[cluster]

        adjoint = solver_prop(model["PropagationFactor"], hidden_gradient, True)
        propagation_gradient = np.sum(adjoint * (model["XTrain"] - model["HTrain"]), axis=1)
        gradient = np.concatenate(
            (propagation_gradient, attention_gradient, head_gradient.ravel(order="F"))
        )
        regularization_scale = np.sqrt(model["RegCoeff"])
        result["Gradient"] = gradient + (2 * regularization_scale) * (
            regularization_scale * model["WeightParam"]
        )
    if not np.isfinite(result["Gradient"]).all():
        raise NeuroFANNError(
            "NeuroFANN:NonfiniteGradient",
            "Backward propagation produced a nonfinite parameter gradient.",
        )
    return result
