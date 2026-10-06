import numpy as np

if __package__:
    from .rand_mt19937 import rand_mt19937
    from .shared import MAXIMUM_SEED, integer_scalar
    from .split_validset import split_validset
else:
    from rand_mt19937 import rand_mt19937
    from shared import MAXIMUM_SEED, integer_scalar
    from split_validset import split_validset


def split_cvindex(dataset, num_iter=100, num_fold=5, seed=1):
    dataset = split_validset(dataset, require_cv=False)
    sample_count = dataset["XData"].shape[1]
    num_iter = integer_scalar(100 if num_iter is None else num_iter, "numIter", 1)
    num_fold = integer_scalar(5 if num_fold is None else num_fold, "numFold", 2, sample_count)
    seed = integer_scalar(1 if seed is None else seed, "seed", 0, MAXIMUM_SEED)
    pattern = 4 * dataset["YabtData"] + 2 * dataset["YmtaData"] + dataset["YwmhData"]
    draws = rand_mt19937(seed, num_iter * sample_count)
    assignment = np.arange(sample_count) % num_fold + 1
    folds = np.zeros((num_iter, sample_count), dtype=np.float64)
    for iteration in range(num_iter):
        keys = draws[iteration * sample_count:(iteration + 1) * sample_count]
        order = np.lexsort((keys, pattern))
        folds[iteration, order] = assignment
    return folds
