import numpy as np

if __package__:
    from .shared import NeuroFANNError
else:
    from shared import NeuroFANNError


def metric_auroc(scores, labels):
    score_array = np.asarray(scores)
    label_array = np.asarray(labels)
    if (
        score_array.dtype.kind not in "iuf"
        or not (score_array.ndim <= 1 or (score_array.ndim == 2 and 1 in score_array.shape))
        or not np.all(np.isfinite(score_array))
    ):
        raise NeuroFANNError("NeuroFANN:InvalidScores", "Scores must be a finite real vector.")
    if (
        label_array.dtype.kind not in "biuf"
        or not (label_array.ndim <= 1 or (label_array.ndim == 2 and 1 in label_array.shape))
        or label_array.size != score_array.size
        or np.any((label_array != 0) & (label_array != 1))
    ):
        raise NeuroFANNError("NeuroFANN:InvalidTargets", "Labels must be a binary vector matching the scores.")
    values = score_array.astype(np.float64).reshape(-1)
    positive = label_array.astype(np.float64).reshape(-1) == 1
    positive_count = int(np.count_nonzero(positive))
    negative_count = positive.size - positive_count
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    starts = np.concatenate(([True], np.diff(sorted_values) != 0))
    first_position = np.flatnonzero(starts) + 1
    last_position = np.concatenate((first_position[1:] - 1, [values.size]))
    group_ranks = (first_position + last_position) / 2
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = group_ranks[np.cumsum(starts) - 1]
    rank_sum = float(np.sum(ranks[positive]))
    return (rank_sum - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count)
