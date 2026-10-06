from time import perf_counter

import numpy as np

if __package__:
    from .loss_measure import loss_measure
    from .model_backward import model_backward
    from .model_forward import model_forward
    from .param_reshape_ import param_reshape_
    from .param_update import copy_optimizer, floating_vector, param_update, validate_optimizer
    from .shared import NeuroFANNError, integer_scalar, logical_flag
else:
    from loss_measure import loss_measure
    from model_backward import model_backward
    from model_forward import model_forward
    from param_reshape_ import param_reshape_
    from param_update import copy_optimizer, floating_vector, param_update, validate_optimizer
    from shared import NeuroFANNError, integer_scalar, logical_flag


def param_training(model):
    trained = dict(model)
    epochs = integer_scalar(model["MaxEpoch"], "MaxEpoch", 1)
    weights = floating_vector(model["WeightParam"], "WeightParam", "NeuroFANN:InvalidWeights")
    if "AdamParam" not in model:
        raise NeuroFANNError(
            "NeuroFANN:InvalidOptimizerState", "Initialize AdamParam with init_adamopt before training."
        )
    trained["AdamParam"] = validate_optimizer(model["AdamParam"], weights.shape)
    history_value = model.get("StoreHistory")
    store_history = False if history_value is None else logical_flag(history_value, "StoreHistory")
    trained["MaxEpoch"] = epochs
    trained["StoreHistory"] = store_history
    trained["WeightParam"] = weights.copy()
    trained["WeightEpoch"] = [None] * epochs if store_history else []
    trained["LossTrain"] = np.full(epochs, np.nan, dtype=np.float64)
    trained["LossValid"] = np.full(epochs, np.nan, dtype=np.float64)
    trained["ObjectiveTrain"] = np.full(epochs, np.nan, dtype=np.float64)
    trained["BestEpoch"] = 0
    trained["BestValidationLoss"] = np.inf
    best_weights = trained["WeightParam"].copy()
    best_adam = copy_optimizer(trained["AdamParam"])
    started = perf_counter()

    for index in range(epochs):
        trained["IdxEpoch"] = index + 1
        if store_history:
            trained["WeightEpoch"][index] = trained["WeightParam"].copy()
        trained = param_reshape_(trained)
        trained = model_forward(trained, include_test=False)
        trained = loss_measure(trained)
        if not all(np.isfinite(trained[name][index]) for name in ("LossTrain", "LossValid", "ObjectiveTrain")):
            raise NeuroFANNError(
                "NeuroFANN:NonfiniteTrainingLoss",
                f"Nonfinite loss at epoch {index + 1}. Check input scaling, regularization, and the learning rate.",
            )
        validation_loss = float(trained["LossValid"][index])
        if validation_loss < trained["BestValidationLoss"]:
            trained["BestEpoch"] = index + 1
            trained["BestValidationLoss"] = validation_loss
            best_weights = trained["WeightParam"].copy()
            best_adam = copy_optimizer(trained["AdamParam"])
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
