import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy

if __package__:
    from .init_adamopt import init_adamopt
    from .model_initialize import model_initialize
    from .model_options import model_options
    from .param_initialize import param_initialize
    from .param_training import param_training
    from .shared import load_dataset
    from .split_validset import split_validset
else:
    from init_adamopt import init_adamopt
    from model_initialize import model_initialize
    from model_options import model_options
    from param_initialize import param_initialize
    from param_training import param_training
    from shared import load_dataset
    from split_validset import split_validset


def main_NeuroFANN(data_source=None, overrides=None):
    dataset = split_validset(load_dataset(data_source))
    parameter = model_options(overrides, dataset)
    iterations = parameter["NumIter"]
    folds = parameter["NumFold"]
    sample_count = dataset["XData"].shape[1]
    test_count = dataset["XTest"].shape[1]
    score_abt = np.empty((iterations, folds, test_count), dtype=np.float64)
    score_mta = np.empty_like(score_abt)
    score_wmh = np.empty_like(score_abt)
    source = "supplied Dataset mapping" if isinstance(data_source, Mapping) else str(
        Path(data_source).expanduser().resolve() if data_source is not None else Path(__file__).with_name("data_sample.mat")
    )
    results = {
        "Parameter": parameter.copy(),
        "DataSource": source,
        "PythonVersion": platform.python_version(),
        "NumPyVersion": np.__version__,
        "SciPyVersion": scipy.__version__,
        "OutOfFoldABT": np.full((iterations, sample_count), np.nan),
        "OutOfFoldMTA": np.full((iterations, sample_count), np.nan),
        "OutOfFoldWMH": np.full((iterations, sample_count), np.nan),
        "Folds": [[None for _ in range(folds)] for _ in range(iterations)],
        "SampleIndexBase": 0,
    }
    started = perf_counter()
    for iteration in range(iterations):
        context = {"IdxIter": iteration + 1, "Dataset": dataset, "Parameter": parameter}
        for fold in range(folds):
            model = param_initialize(model_initialize(context, fold + 1))
            model["AdamParam"] = init_adamopt(model["NumParam"], model["LearnRate"])
            model = param_training(model)
            score_abt[iteration, fold] = model["PabtTest"]
            score_mta[iteration, fold] = model["PmtaTest"]
            score_wmh[iteration, fold] = model["PwmhTest"]
            for target in ("abt", "mta", "wmh"):
                results["OutOfFold" + target.upper()][iteration, model["IdxValid"]] = model["P" + target + "Valid"]
            names = (
                "BestEpoch", "BestValidationLoss", "WeightParam", "SizeParam", "AdamParam",
                "LossTrain", "LossValid", "ObjectiveTrain", "IdxTrain", "IdxValid", "Seed", "TrainingTime",
            )
            summary = {name: model[name] for name in names}
            if parameter["StoreHistory"]:
                summary["WeightEpoch"] = model["WeightEpoch"]
            results["Folds"][iteration][fold] = summary
            if parameter["Verbose"]:
                print(
                    f"Iteration {iteration + 1}/{iterations}, fold {fold + 1}/{folds}: "
                    f"epoch {model['BestEpoch']}, validation loss {model['BestValidationLoss']:.8g}",
                    flush=True,
                )
    results["ElapsedSeconds"] = perf_counter() - started
    return score_abt, score_mta, score_wmh, results


def save_results(path, score_abt, score_mta, score_wmh, results):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    folds = results["Folds"]
    shape = (len(folds), len(folds[0]))
    records = [fold for iteration in folds for fold in iteration]
    assignments = np.zeros_like(results["OutOfFoldABT"], dtype=np.int64)
    for iteration, row in enumerate(folds):
        for fold, record in enumerate(row):
            assignments[iteration, record["IdxValid"]] = fold + 1
    arrays = {
        "ScoreABT": score_abt,
        "ScoreMTA": score_mta,
        "ScoreWMH": score_wmh,
        "OutOfFoldABT": results["OutOfFoldABT"],
        "OutOfFoldMTA": results["OutOfFoldMTA"],
        "OutOfFoldWMH": results["OutOfFoldWMH"],
        "FoldAssignments": assignments,
        "BestEpoch": np.asarray([record["BestEpoch"] for record in records]).reshape(shape),
        "BestValidationLoss": np.asarray([record["BestValidationLoss"] for record in records]).reshape(shape),
        "AdamStep": np.asarray([record["AdamParam"]["t"] for record in records]).reshape(shape),
    }
    for name in ("WeightParam", "LossTrain", "LossValid", "ObjectiveTrain"):
        stacked = np.stack([record[name] for record in records])
        arrays[name] = stacked.reshape(*shape, stacked.shape[-1])
    for name in ("m", "v"):
        stacked = np.stack([record["AdamParam"][name] for record in records])
        arrays["Adam" + name.upper()] = stacked.reshape(*shape, stacked.shape[-1])
    if results["Parameter"]["StoreHistory"]:
        history = np.stack([np.stack(record["WeightEpoch"]) for record in records])
        arrays["WeightEpoch"] = history.reshape(*shape, *history.shape[1:])
    metadata = {key: results[key] for key in (
        "Parameter", "DataSource", "PythonVersion", "NumPyVersion",
        "SciPyVersion", "ElapsedSeconds", "SampleIndexBase",
    )}
    arrays["MetadataJSON"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    with destination.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    return destination


def _cli():
    parser = argparse.ArgumentParser(prog="main_NeuroFANN", description="Train the NeuroFANN Python implementation.")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--folds", type=int)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--regularization", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--store-history", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("neurofann_results.npz"))
    arguments = parser.parse_args()
    options = {
        "MaxEpoch": arguments.epochs, "LearnRate": arguments.learning_rate,
        "RegCoeff": arguments.regularization, "Seed": arguments.seed,
        "StoreHistory": arguments.store_history, "Verbose": arguments.verbose,
    }
    for argument, field in (("iterations", "NumIter"), ("folds", "NumFold")):
        value = getattr(arguments, argument)
        if value is not None:
            options[field] = value
    output = main_NeuroFANN(arguments.data, options)
    destination = save_results(arguments.output, *output)
    print(json.dumps({
        "Status": "completed", "Output": str(destination.resolve()),
        "ScoreShape": list(output[0].shape), "ElapsedSeconds": output[3]["ElapsedSeconds"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    _cli()

