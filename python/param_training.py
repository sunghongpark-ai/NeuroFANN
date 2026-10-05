from copy import deepcopy
from time import perf_counter

import numpy as np

if __package__:
    from .init_adamopt import _positive_integer
    from .loss_measure import loss_measure
    from .model_backward import model_backward
    from .model_forward import model_forward
    from .param_initialize import _history_flag
    from .param_reshape_ import param_reshape_
    from .param_update import _floating_vector, _validate_optimizer, param_update
    from .shared import NeuroFANNError
else:
    from init_adamopt import _positive_integer
    from loss_measure import loss_measure
    from model_backward import model_backward
    from model_forward import model_forward
    from param_initialize import _history_flag
    from param_reshape_ import param_reshape_
    from param_update import _floating_vector, _validate_optimizer, param_update
    from shared import NeuroFANNError


def param_training(model):
    trained = dict(model)
    epochs = _positive_integer(model["MaxEpoch"], "MaxEpoch")
    weights = _floating_vector(model["WeightParam"], "WeightParam", "NeuroFANN:InvalidWeights")
    trained["AdamParam"] = _validate_optimizer(model.get("AdamParam"), weights.shape)
    trained["WeightParam"] = weights.copy()
    trained["StoreHistory"] = _history_flag(model.get("StoreHistory", False))
    trained["WeightEpoch"] = [None] * epochs if trained["StoreHistory"] else []
    trained["LossTrain"] = np.full(epochs, np.nan, dtype=np.float64)
    trained["LossValid"] = np.full(epochs, np.nan, dtype=np.float64)
    trained["ObjectiveTrain"] = np.full(epochs, np.nan, dtype=np.float64)
    trained["BestEpoch"] = 0
    trained["BestValidationLoss"] = np.inf
    best_weights = None
    best_adam = None
    started = perf_counter()

    for index in range(epochs):
        trained["IdxEpoch"] = index + 1
        if trained["StoreHistory"]:
            trained["WeightEpoch"][index] = trained["WeightParam"].copy()
        trained = param_reshape_(trained)
        trained = model_forward(trained, include_test=False)
        trained = loss_measure(trained)
        if not all(np.isfinite(trained[name][index]) for name in ("LossTrain", "LossValid", "ObjectiveTrain")):
            raise NeuroFANNError("NeuroFANN:NonfiniteTrainingLoss", f"Nonfinite loss at epoch {index + 1}. Check input scaling, regularization, and learning rate.")
        validation_loss = float(trained["LossValid"][index])
        if validation_loss < trained["BestValidationLoss"]:
            trained["BestEpoch"] = index + 1
            trained["BestValidationLoss"] = validation_loss
            best_weights = trained["WeightParam"].copy()
            best_adam = deepcopy(trained["AdamParam"])
        if index + 1 < epochs:
            trained = model_backward(trained)
            trained = param_update(trained)

    trained["WeightParam"] = best_weights
    trained["AdamParam"] = best_adam
    trained = param_reshape_(trained)
    trained = model_forward(trained, include_test=True)
    trained = model_backward(trained)
    trained["TrainingTime"] = perf_counter() - started
    return trained
