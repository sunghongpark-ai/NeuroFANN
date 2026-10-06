import numpy as np

if __package__:
    from .shared import MAXIMUM_SEED, integer_scalar
else:
    from shared import MAXIMUM_SEED, integer_scalar


def rand_mt19937(seed, count):
    seed = integer_scalar(seed, "Seed", 0, MAXIMUM_SEED)
    count = integer_scalar(count, "The number of draws", 0)
    stream = np.random.RandomState(5489 if seed == 0 else seed)
    return stream.random_sample(count)
