import re

import numpy as np

if __package__:
    from .ppi_laplacian import ppi_laplacian
    from .shared import RESERVED_COLUMNS, NeuroFANNError, format_number, is_csv_token, text_path
    from .split_validset import split_validset
else:
    from ppi_laplacian import ppi_laplacian
    from shared import RESERVED_COLUMNS, NeuroFANNError, format_number, is_csv_token, text_path
    from split_validset import split_validset

FOLD_PATTERN = re.compile(r"cv[0-9]+")


def _invalid(message):
    raise NeuroFANNError("NeuroFANN:InvalidDataset", message)


def _number_fields(values):
    return [format_number(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)]


def data_writecsv(dataset, file):
    path = text_path(file)
    dataset = split_validset(dataset, require_cv=False)
    if dataset["Wppi"].size == 0:
        _invalid("Writing a CSV dataset requires the PPI weights Wppi.")
    if not np.array_equal(ppi_laplacian(dataset["Wppi"]), dataset["Lppi"]):
        _invalid("Lppi must equal the normalized Laplacian of Wppi because the CSV format stores Wppi only.")
    protein_names = [str(name) for name in dataset["IdxProtein"]]
    for name in protein_names:
        if not is_csv_token(name):
            _invalid(f'Protein name "{name}" cannot be written as a plain CSV field.')
        if name in RESERVED_COLUMNS or FOLD_PATTERN.fullmatch(name):
            _invalid(f'Protein name "{name}" is reserved by the CSV format.')
    sample_ids = [str(name) for name in dataset["IdxSampleData"]] + [str(name) for name in dataset["IdxSampleTest"]]
    for name in sample_ids:
        if not is_csv_token(name):
            _invalid(f'Sample id "{name}" cannot be written as a plain CSV field.')
    fold_count = dataset["CVindex"].shape[0]
    width = max(3, len(str(fold_count)))
    fold_names = [f"cv{index:0{width}d}" for index in range(1, fold_count + 1)]
    label_data = np.vstack([dataset["YabtData"], dataset["YmtaData"], dataset["YwmhData"]])
    label_test = [dataset["YabtTest"], dataset["YmtaTest"], dataset["YwmhTest"]]
    empty_folds = [""] * fold_count
    empty_labels = [""] * 3

    lines = [",".join(list(RESERVED_COLUMNS) + fold_names + protein_names)]
    lines.append(",".join(
        ["cluster", "cluster", ""] + empty_labels + empty_folds + _number_fields(dataset["IdxCluster"])
    ))
    for protein, name in enumerate(protein_names):
        lines.append(",".join(
            ["network", name, ""] + empty_labels + empty_folds + _number_fields(dataset["Wppi"][protein])
        ))
    for sample, identifier in enumerate(dataset["IdxSampleData"]):
        lines.append(",".join(
            ["sample", str(identifier), "train"] + _number_fields(label_data[:, sample])
            + _number_fields(dataset["CVindex"][:, sample]) + _number_fields(dataset["XData"][:, sample])
        ))
    for sample, identifier in enumerate(dataset["IdxSampleTest"]):
        labels = [format_number(values[sample]) if values.size else "" for values in label_test]
        lines.append(",".join(
            ["sample", str(identifier), "test"] + labels + empty_folds + _number_fields(dataset["XTest"][:, sample])
        ))
    try:
        with open(path, "w", encoding="ascii", newline="\n") as stream:
            stream.write("\n".join(lines) + "\n")
    except OSError as exception:
        raise NeuroFANNError("NeuroFANN:OutputFailure", f"Cannot create {path}: {exception}") from exception
