import math

import numpy as np

if __package__:
    from .metric_auroc import metric_auroc
    from .shared import NeuroFANNError
    from .split_validset import split_validset
else:
    from metric_auroc import metric_auroc
    from shared import NeuroFANNError
    from split_validset import split_validset


def _row_statistic(values, kind):
    result = np.full(values.shape[0], np.nan)
    for row in range(values.shape[0]):
        entries = values[row]
        count = entries.size
        if count == 0 or np.any(np.isnan(entries)):
            continue
        total = 0.0
        for entry in entries:
            total = total + float(entry)
        average = total / count
        if kind == "mean":
            result[row] = average
        elif kind == "std":
            if count > 1:
                squares = 0.0
                for entry in entries:
                    deviation = float(entry) - average
                    squares = squares + deviation * deviation
                result[row] = math.sqrt(squares / (count - 1))
        elif kind == "min":
            result[row] = float(np.min(entries))
        elif kind == "max":
            result[row] = float(np.max(entries))
    return result


def model_evaluate(score_abt, score_mta, score_wmh, results, dataset):
    dataset = split_validset(dataset, require_cv=False)
    test_count = dataset["XTest"].shape[1]
    sample_count = dataset["XData"].shape[1]
    score_sets = []
    for scores in (score_abt, score_mta, score_wmh):
        array = np.asarray(scores)
        if (
            array.ndim != 3 or array.dtype.kind not in "iuf" or array.shape[0] < 1 or array.shape[1] < 1
            or array.shape[2] != test_count or not np.all(np.isfinite(array))
            or np.any((array < 0) | (array > 1))
        ):
            raise NeuroFANNError(
                "NeuroFANN:InvalidScores",
                "Each score set must be a NumIter x NumFold x NumTest array of probabilities.",
            )
        score_sets.append(array.astype(np.float64))
    iterations, folds = score_sets[0].shape[:2]
    if any(array.shape != score_sets[0].shape for array in score_sets):
        raise NeuroFANNError("NeuroFANN:InvalidScores", "Score sets must have equal sizes.")
    out_of_fold = []
    for name in ("OutOfFoldABT", "OutOfFoldMTA", "OutOfFoldWMH"):
        array = np.asarray(results[name])
        if array.dtype.kind not in "iuf" or array.shape != (iterations, sample_count):
            raise NeuroFANNError("NeuroFANN:InvalidScores", "Out-of-fold scores must be NumIter-by-NumData matrices.")
        out_of_fold.append(array.astype(np.float64))
    labels_data = np.vstack([dataset["YabtData"], dataset["YmtaData"], dataset["YwmhData"]])
    labels_test = np.full((3, test_count), np.nan)
    available = []
    for target, name in enumerate(("YabtTest", "YmtaTest", "YwmhTest")):
        present = dataset[name].size > 0
        available.append(present)
        if present:
            labels_test[target] = dataset[name]
    evaluation = {
        "Targets": ["ABT", "MTA", "WMH"],
        "NumModels": iterations * folds,
        "IdxSampleData": dataset["IdxSampleData"].copy(),
        "IdxSampleTest": dataset["IdxSampleTest"].copy(),
        "LabelsData": labels_data,
        "LabelsTest": labels_test,
        "TestAUROC": np.full((iterations, folds, 3), np.nan),
        "EnsembleTestScore": np.zeros((3, test_count)),
        "EnsembleTestAUROC": np.full(3, np.nan),
        "OutOfFoldAUROC": np.full((iterations, 3), np.nan),
        "MeanOutOfFoldScore": np.zeros((3, sample_count)),
    }
    for target in range(3):
        scores = score_sets[target]
        total = np.zeros(test_count)
        for iteration in range(iterations):
            for fold in range(folds):
                current = scores[iteration, fold]
                total = total + current
                if available[target]:
                    evaluation["TestAUROC"][iteration, fold, target] = metric_auroc(current, labels_test[target])
        evaluation["EnsembleTestScore"][target] = total / (iterations * folds)
        if available[target] and test_count > 0:
            evaluation["EnsembleTestAUROC"][target] = metric_auroc(
                evaluation["EnsembleTestScore"][target], labels_test[target]
            )
        total = np.zeros(sample_count)
        complete = True
        for iteration in range(iterations):
            current = out_of_fold[target][iteration]
            if np.all(np.isfinite(current)):
                evaluation["OutOfFoldAUROC"][iteration, target] = metric_auroc(current, labels_data[target])
            else:
                complete = False
            total = total + current
        evaluation["MeanOutOfFoldScore"][target] = total / iterations if complete else np.nan
    test_values = np.vstack([evaluation["TestAUROC"][:, :, target].reshape(-1) for target in range(3)])
    evaluation["TestAUROCMean"] = _row_statistic(test_values, "mean")
    evaluation["TestAUROCStd"] = _row_statistic(test_values, "std")
    evaluation["TestAUROCMin"] = _row_statistic(test_values, "min")
    evaluation["TestAUROCMax"] = _row_statistic(test_values, "max")
    evaluation["TestAUROCOverall"] = float(_row_statistic(evaluation["TestAUROCMean"][None, :], "mean")[0])
    out_of_fold_values = evaluation["OutOfFoldAUROC"].T
    evaluation["OutOfFoldAUROCMean"] = _row_statistic(out_of_fold_values, "mean")
    evaluation["OutOfFoldAUROCStd"] = _row_statistic(out_of_fold_values, "std")
    evaluation["OutOfFoldAUROCMin"] = _row_statistic(out_of_fold_values, "min")
    evaluation["OutOfFoldAUROCMax"] = _row_statistic(out_of_fold_values, "max")
    return evaluation
