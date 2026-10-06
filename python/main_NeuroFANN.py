import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy
from scipy.io import loadmat

if __package__:
    from .data_readcsv import data_readcsv
    from .data_writeresults import data_writeresults
    from .init_adamopt import init_adamopt
    from .model_evaluate import model_evaluate
    from .model_initialize import model_initialize
    from .model_options import model_options
    from .param_initialize import param_initialize
    from .param_training import param_training
    from .shared import NeuroFANNError, TARGETS, default_dataset_path
    from .split_validset import split_validset
else:
    from data_readcsv import data_readcsv
    from data_writeresults import data_writeresults
    from init_adamopt import init_adamopt
    from model_evaluate import model_evaluate
    from model_initialize import model_initialize
    from model_options import model_options
    from param_initialize import param_initialize
    from param_training import param_training
    from shared import NeuroFANNError, TARGETS, default_dataset_path
    from split_validset import split_validset

SUMMARY_FIELDS = (
    "BestEpoch", "BestValidationLoss", "WeightParam", "SizeParam", "AdamParam",
    "LossTrain", "LossValid", "ObjectiveTrain", "IdxTrain", "IdxValid", "Seed", "TrainingTime",
)


def _fixed(value):
    number = float(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "Inf" if number > 0 else "-Inf"
    return f"{number:.4f}"


def _matlab_text(value):
    if type(value).__name__ == "MatlabOpaque":
        raise NeuroFANNError(
            "NeuroFANN:UnsupportedMetadata",
            "Save text identifiers as cell arrays of character vectors (MAT v7) for Python.",
        )
    array = np.asarray(value)
    if array.dtype.kind == "U":
        return np.asarray([text.rstrip() for text in array.ravel(order="F")], dtype=object)
    if array.dtype.kind != "O":
        return array
    names = []
    for item in array.ravel(order="F"):
        text = np.asarray(item)
        if text.dtype.kind not in "US":
            raise NeuroFANNError("NeuroFANN:InvalidDataset", "Identifiers must be text or numeric values.")
        names.append("".join(text.ravel().tolist()))
    return np.asarray(names, dtype=object)


def load_dataset(data_source=None):
    if isinstance(data_source, Mapping):
        return dict(data_source), "supplied Dataset mapping"
    if data_source is None:
        data_source = default_dataset_path()
    if not isinstance(data_source, (str, Path)) or str(data_source) == "":
        raise NeuroFANNError(
            "NeuroFANN:InvalidDataset", "Provide a Dataset mapping, a CSV file path, or a MAT-file path."
        )
    path = Path(data_source).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return data_readcsv(path), str(path)
    if suffix not in (".mat", ".npz"):
        raise NeuroFANNError("NeuroFANN:UnsupportedDataFile", f"Use a .csv, .mat, or .npz dataset file: {path}.")
    if not path.is_file():
        raise NeuroFANNError("NeuroFANN:MissingDataFile", f"Dataset file does not exist: {path}.")
    if suffix == ".npz":
        try:
            with np.load(path, allow_pickle=False) as archive:
                return {key: archive[key].copy() for key in archive.files}, str(path)
        except Exception as exception:
            raise NeuroFANNError("NeuroFANN:InvalidDataset", f"Cannot read {path}: {exception}") from exception
    try:
        payload = loadmat(
            path, variable_names=["Dataset"], struct_as_record=False, squeeze_me=False, chars_as_strings=True
        )
    except NotImplementedError as exception:
        raise NeuroFANNError(
            "NeuroFANN:UnsupportedDataFile", "Use MAT v7 or CSV; MAT v7.3 is not supported."
        ) from exception
    except Exception as exception:
        raise NeuroFANNError("NeuroFANN:InvalidDataset", f"Cannot read {path}: {exception}") from exception
    structure = payload.get("Dataset")
    if not isinstance(structure, np.ndarray) or structure.size != 1 or not hasattr(structure.item(), "_fieldnames"):
        raise NeuroFANNError("NeuroFANN:InvalidDataset", "The MAT file must contain a scalar struct named Dataset.")
    structure = structure.item()
    dataset = {name: getattr(structure, name) for name in structure._fieldnames}
    for name in ("IdxProtein", "IdxSampleData", "IdxSampleTest"):
        if name in dataset:
            dataset[name] = _matlab_text(dataset[name])
    return dataset, str(path)


def main_NeuroFANN(data_source=None, overrides=None):
    raw_dataset, source = load_dataset(data_source)
    dataset = split_validset(raw_dataset)
    parameter = model_options(overrides, dataset)
    iterations = parameter["NumIter"]
    folds = parameter["NumFold"]
    sample_count = dataset["XData"].shape[1]
    test_count = dataset["XTest"].shape[1]
    score_abt = np.empty((iterations, folds, test_count), dtype=np.float64)
    score_mta = np.empty_like(score_abt)
    score_wmh = np.empty_like(score_abt)
    results = {
        "Parameter": dict(parameter),
        "DataSource": source,
        "Runtime": f"Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}",
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
            for target in TARGETS:
                results["OutOfFold" + target.upper()][iteration, model["IdxValid"]] = model["P" + target + "Valid"]
            summary = {name: model[name] for name in SUMMARY_FIELDS}
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
    results["Evaluation"] = model_evaluate(score_abt, score_mta, score_wmh, results, dataset)
    if parameter["Verbose"]:
        evaluation = results["Evaluation"]
        print("Test AUROC, mean of {} models: ABT {}, MTA {}, WMH {}".format(
            iterations * folds, *map(_fixed, evaluation["TestAUROCMean"])), flush=True)
        print("Test AUROC, ensemble mean risk: ABT {}, MTA {}, WMH {}".format(
            *map(_fixed, evaluation["EnsembleTestAUROC"])), flush=True)
        print("Out-of-fold AUROC, mean of {} iterations: ABT {}, MTA {}, WMH {}".format(
            iterations, *map(_fixed, evaluation["OutOfFoldAUROCMean"])), flush=True)
    return score_abt, score_mta, score_wmh, results


def save_results(path, score_abt, score_mta, score_wmh, results):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    folds = results["Folds"]
    shape = (len(folds), len(folds[0]))
    records = [fold for iteration in folds for fold in iteration]
    assignments = np.zeros(results["OutOfFoldABT"].shape, dtype=np.int64)
    for iteration, row in enumerate(folds):
        for fold, record in enumerate(row):
            assignments[iteration, record["IdxValid"]] = fold + 1
    evaluation = results["Evaluation"]
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
        "TestAUROC": evaluation["TestAUROC"],
        "EnsembleTestScore": evaluation["EnsembleTestScore"],
        "EnsembleTestAUROC": evaluation["EnsembleTestAUROC"],
        "OutOfFoldAUROC": evaluation["OutOfFoldAUROC"],
        "MeanOutOfFoldScore": evaluation["MeanOutOfFoldScore"],
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
    names = ("Parameter", "DataSource", "Runtime", "ElapsedSeconds", "SampleIndexBase")
    metadata = {key: results[key] for key in names}
    arrays["MetadataJSON"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    with destination.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    return destination


def _cli(argv=None):
    parser = argparse.ArgumentParser(prog="main_NeuroFANN", description="Train and evaluate NeuroFANN.")
    parser.add_argument(
        "--data", type=Path, default=None, help="dataset CSV, MAT, or NPZ file (default: dataset/sample.csv)"
    )
    parser.add_argument("--iterations", type=int, help="split iterations, NumIter (default: every CVindex row)")
    parser.add_argument("--folds", type=int, help="folds per iteration, NumFold (default: largest fold ID)")
    parser.add_argument("--epochs", type=int, default=500, help="candidate epochs per fold, MaxEpoch (default: 500)")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam step size, LearnRate (default: 0.001)")
    parser.add_argument("--regularization", type=float, default=0.005, help="L2 coefficient, RegCoeff (default: 0.005)")
    parser.add_argument("--seed", type=int, default=1, help="seed of the first iteration, Seed (default: 1)")
    parser.add_argument("--store-history", action="store_true", help="keep the weights of every epoch")
    parser.add_argument("--verbose", action="store_true", help="print one line per fold and an AUROC summary")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("neurofann_results"),
        help="folder for the CSV and JSON result files (default: neurofann_results)",
    )
    parser.add_argument("--output", type=Path, default=None, help="optional compressed NPZ archive of all fold states")
    arguments = parser.parse_args(argv)
    options = {
        "MaxEpoch": arguments.epochs, "LearnRate": arguments.learning_rate,
        "RegCoeff": arguments.regularization, "Seed": arguments.seed,
        "StoreHistory": arguments.store_history, "Verbose": arguments.verbose,
    }
    for argument, field in (("iterations", "NumIter"), ("folds", "NumFold")):
        value = getattr(arguments, argument)
        if value is not None:
            options[field] = value
    try:
        output = main_NeuroFANN(arguments.data, options)
        files = data_writeresults(output[3], arguments.output_dir)
        if arguments.output is not None:
            files.append(save_results(arguments.output, *output))
    except NeuroFANNError as exception:
        print(f"main_NeuroFANN: error: {exception}", file=sys.stderr)
        return 1
    except OSError as exception:
        print(f"main_NeuroFANN: error: NeuroFANN:OutputFailure: {exception}", file=sys.stderr)
        return 1
    evaluation = output[3]["Evaluation"]
    print(json.dumps({
        "Status": "completed",
        "Outputs": [str(Path(file).resolve()) for file in files],
        "ScoreShape": list(output[0].shape),
        "TestAUROCMean": dict(zip(evaluation["Targets"], map(float, evaluation["TestAUROCMean"]))),
        "EnsembleTestAUROC": dict(zip(evaluation["Targets"], map(float, evaluation["EnsembleTestAUROC"]))),
        "ElapsedSeconds": output[3]["ElapsedSeconds"],
    }, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
