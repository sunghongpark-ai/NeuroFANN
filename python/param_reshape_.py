import numpy as np
from scipy import sparse

if __package__:
    from .shared import NeuroFANNError
else:
    from shared import NeuroFANNError


def param_reshape_(model):
    result = dict(model)
    protein_count = model["NumProtein"]
    cluster_count = model["NumCluster"]
    expected_sizes = np.array(
        [[protein_count, 1], [protein_count, 1]] + [[cluster_count, 1]] * 3,
        dtype=np.int64,
    )
    raw_weights = np.asarray(model["WeightParam"])
    if (
        not np.array_equal(np.asarray(model["SizeParam"]), expected_sizes)
        or raw_weights.ndim != 1
        or raw_weights.size != int(np.prod(expected_sizes, axis=1).sum())
    ):
        raise NeuroFANNError(
            "NeuroFANN:InvalidParameterSize",
            "The parameter vector and parameter shapes do not match the model.",
        )
    if raw_weights.dtype.kind != "f" or not np.all(np.isfinite(raw_weights)):
        raise NeuroFANNError(
            "NeuroFANN:InvalidParameters",
            "WeightParam must contain finite real floating-point values.",
        )
    weights = np.array(raw_weights, dtype=np.float64, copy=True)
    result["WeightParam"] = weights
    offset = 0
    for name, count in zip(("Uprot", "Aclus", "Babt", "Bmta", "Bwmh"), expected_sizes[:, 0]):
        result[name] = weights[offset:offset + count].copy()
        offset += count
    result["Udiag"] = np.diag(result["Uprot"])

    cluster_index = np.asarray(model["IdxCluster"]).reshape(-1)
    members = model.get("ClusterMembers")
    assignment = model.get("ClusterAssignment")
    if (
        "ClusterIndexSnapshot" not in model
        or not np.array_equal(model["ClusterIndexSnapshot"], cluster_index)
        or members is None
        or len(members) != cluster_count
        or assignment is None
        or not sparse.issparse(assignment)
        or assignment.shape != (cluster_count, protein_count)
    ):
        if (
            cluster_index.dtype.kind not in "iuf"
            or cluster_index.size != protein_count
            or not np.all(np.isfinite(cluster_index))
            or np.any(cluster_index != np.trunc(cluster_index))
            or np.any((cluster_index < 1) | (cluster_index > cluster_count))
        ):
            raise NeuroFANNError(
                "NeuroFANN:InvalidClusters",
                "IdxCluster must assign one integer cluster index to every protein.",
            )
        members = [np.flatnonzero(cluster_index == cluster) for cluster in range(1, cluster_count + 1)]
        result["ClusterMembers"] = members
        result["ClusterAssignment"] = sparse.csr_matrix(
            (
                np.ones(protein_count, dtype=np.float64),
                (cluster_index.astype(np.int64) - 1, np.arange(protein_count)),
            ),
            shape=(cluster_count, protein_count),
        )
        result["ClusterIndexSnapshot"] = cluster_index.copy()

    probabilities = np.zeros(protein_count, dtype=np.float64)
    attention = result["Aclus"]
    for indices in members:
        if len(indices) == 0:
            raise NeuroFANNError("NeuroFANN:EmptyCluster", "Every cluster must contain a protein.")
        logits = attention[indices]
        with np.errstate(over="ignore", under="ignore"):
            unnormalized = np.exp(logits - np.max(logits))
        probabilities[indices] = unnormalized / unnormalized.sum()
    result["Aprob"] = probabilities
    result.pop("PropagationFactor", None)
    return result
