from collections.abc import Mapping
import json
import math
from pathlib import Path

import numpy as np

if __package__:
    from .shared import NeuroFANNError, TARGETS, format_number
else:
    from shared import NeuroFANNError, TARGETS, format_number


def _cell(value):
    number = float(value)
    return "" if math.isnan(number) else format_number(number)


def _write_table(path, header, rows):
    lines = [",".join(header)] + [",".join(row) for row in rows]
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(lines) + "\n")
    except OSError as exception:
        raise NeuroFANNError("NeuroFANN:OutputFailure", f"Cannot create {path}: {exception}") from exception


def _json_value(value):
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else number
    return value


def data_writeresults(results, output_directory):
    if not isinstance(results, Mapping) or "Evaluation" not in results or "Parameter" not in results:
        raise NeuroFANNError("NeuroFANN:InvalidResults", "Results must be the fourth output of main_NeuroFANN.")
    if not isinstance(output_directory, (str, Path)) or str(output_directory) == "":
        raise NeuroFANNError("NeuroFANN:InvalidResults", "Provide the output folder as text.")
    directory = Path(output_directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exception:
        raise NeuroFANNError("NeuroFANN:OutputFailure", f"Cannot create {directory}: {exception}") from exception
    evaluation = results["Evaluation"]
    label_header = ["y_" + target for target in TARGETS]
    risk_header = ["risk_" + target for target in TARGETS]
    files = []

    path = directory / "predictions_test.csv"
    rows = [
        [str(identifier)] + [_cell(value) for value in evaluation["LabelsTest"][:, index]]
        + [_cell(value) for value in evaluation["EnsembleTestScore"][:, index]]
        for index, identifier in enumerate(evaluation["IdxSampleTest"])
    ]
    _write_table(path, ["id"] + label_header + risk_header, rows)
    files.append(path)

    path = directory / "predictions_oof.csv"
    rows = [
        [str(identifier)] + [_cell(value) for value in evaluation["LabelsData"][:, index]]
        + [_cell(value) for value in evaluation["MeanOutOfFoldScore"][:, index]]
        for index, identifier in enumerate(evaluation["IdxSampleData"])
    ]
    _write_table(path, ["id"] + label_header + risk_header, rows)
    files.append(path)

    path = directory / "auroc_models.csv"
    iterations, folds = evaluation["TestAUROC"].shape[:2]
    rows = [
        [_cell(iteration + 1), _cell(fold + 1)] + [_cell(value) for value in evaluation["TestAUROC"][iteration, fold]]
        for iteration in range(iterations)
        for fold in range(folds)
    ]
    _write_table(path, ["iteration", "fold"] + ["auroc_" + target for target in TARGETS], rows)
    files.append(path)

    path = directory / "auroc_summary.csv"
    columns = (
        "TestAUROCMean", "TestAUROCStd", "TestAUROCMin", "TestAUROCMax", "EnsembleTestAUROC",
        "OutOfFoldAUROCMean", "OutOfFoldAUROCStd", "OutOfFoldAUROCMin", "OutOfFoldAUROCMax",
    )
    rows = [
        [target.upper()] + [_cell(evaluation[column][index]) for column in columns]
        for index, target in enumerate(TARGETS)
    ]
    _write_table(path, [
        "target", "test_mean", "test_std", "test_min", "test_max", "test_ensemble",
        "oof_mean", "oof_std", "oof_min", "oof_max",
    ], rows)
    files.append(path)

    report = {
        "DataSource": results["DataSource"],
        "Runtime": results["Runtime"],
        "Parameter": results["Parameter"],
        "NumModels": evaluation["NumModels"],
        "NumTrain": len(evaluation["IdxSampleData"]),
        "NumTest": len(evaluation["IdxSampleTest"]),
        "ElapsedSeconds": results["ElapsedSeconds"],
        "Targets": evaluation["Targets"],
        "TestAUROCMean": evaluation["TestAUROCMean"],
        "TestAUROCStd": evaluation["TestAUROCStd"],
        "TestAUROCOverall": evaluation["TestAUROCOverall"],
        "EnsembleTestAUROC": evaluation["EnsembleTestAUROC"],
        "OutOfFoldAUROCMean": evaluation["OutOfFoldAUROCMean"],
        "OutOfFoldAUROCStd": evaluation["OutOfFoldAUROCStd"],
    }
    path = directory / "run_summary.json"
    try:
        path.write_text(json.dumps(_json_value(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError as exception:
        raise NeuroFANNError("NeuroFANN:OutputFailure", f"Cannot create {path}: {exception}") from exception
    files.append(path)
    return files
