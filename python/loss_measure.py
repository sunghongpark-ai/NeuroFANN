import numpy as np

if __package__:
    from .shared import NeuroFANNError
else:
    from shared import NeuroFANNError


def _updated_history(model, name, epoch, value):
    existing = np.asarray(model.get(name, np.empty(0)), dtype=np.float64).reshape(-1)
    history = np.full(max(existing.size, epoch), np.nan, dtype=np.float64)
    history[:existing.size] = existing
    history[epoch - 1] = value
    return history


def _validated_targets(model, split, count, logits):
    labels = [np.asarray(model[f"{name}{split}"]).reshape(-1) for name in ("Yabt", "Ymta", "Ywmh")]
    if (
        count < 1
        or logits.shape != (3, count)
        or logits.dtype.kind not in "biuf"
        or not np.all(np.isfinite(logits))
        or any(
            label.size != count
            or label.dtype.kind not in "biuf"
            or not np.all(np.isfinite(label))
            or np.any((label < 0) | (label > 1))
            for label in labels
        )
    ):
        raise NeuroFANNError(
            "NeuroFANN:InvalidTargets",
            "Labels and finite logits must have matching shapes, with labels in [0, 1].",
        )
    return np.vstack(labels).astype(np.float64, copy=False)


def loss_measure(model):
    epoch_value = np.asarray(model["IdxEpoch"])
    regularization = np.asarray(model["RegCoeff"])
    if (
        epoch_value.size != 1
        or epoch_value.dtype.kind not in "iuf"
        or not np.isfinite(epoch_value).all()
        or epoch_value.reshape(-1)[0] < 1
        or epoch_value.reshape(-1)[0] != np.trunc(epoch_value.reshape(-1)[0])
    ):
        raise NeuroFANNError("NeuroFANN:InvalidParameter", "IdxEpoch must be a positive integer.")
    if (
        regularization.size != 1
        or regularization.dtype.kind not in "iuf"
        or not np.isfinite(regularization).all()
        or regularization.reshape(-1)[0] < 0
    ):
        raise NeuroFANNError("NeuroFANN:InvalidParameter", "RegCoeff must be finite and nonnegative.")
    epoch = int(epoch_value.reshape(-1)[0])
    regularization = regularization.reshape(-1)[0]
    result = dict(model)
    for split in ("Train", "Valid"):
        logits = np.asarray(model[f"Logits{split}"])
        count = model[f"Num{split}"]
        targets = _validated_targets(model, split, count, logits)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            terms = np.maximum(logits, 0) - targets * logits + np.log1p(np.exp(-np.abs(logits)))
            loss = np.sum(terms / count)
        if not np.isfinite(loss):
            raise NeuroFANNError("NeuroFANN:NonfiniteLoss", "The loss exceeds the finite numerical range.")
        result[f"Loss{split}"] = _updated_history(model, f"Loss{split}", epoch, loss)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        if regularization == 0:
            penalty = 0.0
        else:
            scaled_weights = np.sqrt(regularization) * model["WeightParam"]
            penalty = np.sum(scaled_weights**2)
        objective = result["LossTrain"][epoch - 1] + penalty
    if not np.isfinite(objective):
        raise NeuroFANNError(
            "NeuroFANN:NonfiniteLoss", "The regularized objective exceeds the finite numerical range."
        )
    result["ObjectiveTrain"] = _updated_history(model, "ObjectiveTrain", epoch, objective)
    return result
