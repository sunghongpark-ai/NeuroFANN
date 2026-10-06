import re

import numpy as np

if __package__:
    from .ppi_laplacian import ppi_laplacian
    from .shared import RESERVED_COLUMNS, NeuroFANNError, text_path
    from .split_validset import split_validset
else:
    from ppi_laplacian import ppi_laplacian
    from shared import RESERVED_COLUMNS, NeuroFANNError, text_path
    from split_validset import split_validset

NUMBER_PATTERN = re.compile(r"[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?")
FOLD_PATTERN = re.compile(r"cv([0-9]+)")
ALLOWED_BYTES = frozenset([9, 10, 13] + list(range(32, 127)))


def _invalid(message):
    raise NeuroFANNError("NeuroFANN:InvalidCSV", message)


def _first_column_major(mask):
    positions = np.argwhere(np.asarray(mask, dtype=bool).T)
    column, row = positions[0]
    return int(row), int(column)


def _parse_numbers(texts, lines, names, allow_empty):
    texts = np.asarray(texts, dtype=object)
    values = np.zeros(texts.shape, dtype=np.float64)
    if texts.size == 0:
        return values
    empty = np.vectorize(lambda text: text == "", otypes=[bool])(texts)
    invalid = np.vectorize(lambda text: NUMBER_PATTERN.fullmatch(text) is None, otypes=[bool])(texts)
    if allow_empty:
        invalid &= ~empty
    if invalid.any():
        row, column = _first_column_major(invalid)
        _invalid(f'Line {lines[row]}, column {names[column]}: "{texts[row, column]}" is not a finite decimal number.')
    for (row, column), text in np.ndenumerate(texts):
        values[row, column] = np.nan if text == "" else float(text)
    if np.isinf(values).any():
        row, column = _first_column_major(np.isinf(values))
        _invalid(f"Line {lines[row]}, column {names[column]}: the value overflows double precision.")
    return values


def _require_empty(cells, rows, columns, lines, header):
    if len(rows) == 0 or len(columns) == 0:
        return
    filled = np.array([[cells[row][column] != "" for column in columns] for row in rows], dtype=bool)
    if filled.any():
        row, column = _first_column_major(filled)
        _invalid(f"Line {lines[rows[row]]}: column {header[columns[column]]} must be empty for this record.")


def data_readcsv(file):
    path = text_path(file)
    if not path.is_file():
        raise NeuroFANNError("NeuroFANN:MissingDataFile", f"Dataset file does not exist: {path}.")
    try:
        raw = path.read_bytes()
    except OSError as exception:
        raise NeuroFANNError("NeuroFANN:MissingDataFile", f"Cannot open {path}: {exception}") from exception
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    if not ALLOWED_BYTES.issuperset(raw):
        _invalid("The CSV file must contain printable ASCII text only.")
    text = raw.decode("ascii")
    if '"' in text:
        _invalid("Quoted CSV fields are not supported.")
    text = text.replace("\r", "")
    numbered = [
        (number, line) for number, line in enumerate(text.split("\n"), start=1) if line.strip(" \t") != ""
    ]
    if len(numbered) < 2:
        _invalid("The CSV file needs a header and at least one record.")
    header = [field.strip(" \t") for field in numbered[0][1].split(",")]
    column_count = len(header)
    row_lines = [number for number, _ in numbered[1:]]
    cells = []
    for number, line in numbered[1:]:
        fields = [field.strip(" \t") for field in line.split(",")]
        if len(fields) != column_count:
            _invalid(f"Line {number} has {len(fields)} fields; the header has {column_count}.")
        cells.append(fields)

    if any(name == "" for name in header):
        _invalid("Every header field must have a name.")
    if len(set(header)) != column_count:
        _invalid("Header names must be unique.")
    missing = [name for name in RESERVED_COLUMNS if name not in header]
    if missing:
        _invalid("The header is missing: " + ", ".join(missing) + ".")
    record_column, id_column, split_column = (header.index(name) for name in RESERVED_COLUMNS[:3])
    label_columns = [header.index(name) for name in RESERVED_COLUMNS[3:]]
    fold_entries = []
    for column, name in enumerate(header):
        match = FOLD_PATTERN.fullmatch(name)
        if match:
            fold_entries.append((float(match.group(1)), column))
    fold_numbers = sorted(number for number, _ in fold_entries)
    fold_columns = [column for _, column in sorted(fold_entries, key=lambda entry: entry[0])]
    if fold_numbers != [float(number) for number in range(1, len(fold_numbers) + 1)]:
        _invalid("Cross-validation columns must be cv1 to cvK without gaps or duplicates.")
    fold_set = set(fold_columns)
    protein_columns = [
        column for column, name in enumerate(header) if name not in RESERVED_COLUMNS and column not in fold_set
    ]
    if not protein_columns:
        _invalid("The header must contain at least one protein column.")
    protein_names = [header[column] for column in protein_columns]
    protein_count = len(protein_columns)
    auxiliary_columns = [split_column] + label_columns + fold_columns

    records = [row[record_column] for row in cells]
    for row, record in enumerate(records):
        if record not in ("cluster", "network", "sample"):
            _invalid(f'Line {row_lines[row]} has an unknown record type "{record}".')
    cluster_rows = [row for row, record in enumerate(records) if record == "cluster"]
    network_rows = [row for row, record in enumerate(records) if record == "network"]
    sample_rows = [row for row, record in enumerate(records) if record == "sample"]
    _require_empty(cells, cluster_rows + network_rows, auxiliary_columns, row_lines, header)

    if len(cluster_rows) != 1:
        _invalid("The CSV file must contain exactly one cluster record.")
    cluster_row = cluster_rows[0]
    cluster_index = _parse_numbers(
        [[cells[cluster_row][column] for column in protein_columns]], [row_lines[cluster_row]], protein_names, False
    ).reshape(-1)
    if np.any((cluster_index < 1) | (cluster_index != np.trunc(cluster_index))):
        _invalid(f"Line {row_lines[cluster_row]}: cluster IDs must be positive integers.")
    cluster_count = int(np.max(cluster_index))
    if cluster_count > protein_count or np.any(
        np.bincount(cluster_index.astype(np.int64), minlength=cluster_count + 1)[1:] == 0
    ):
        _invalid("Cluster IDs must use every integer from 1 to the largest ID.")

    network_ids = [cells[row][id_column] for row in network_rows]
    positions = [protein_names.index(name) if name in protein_names else -1 for name in network_ids]
    if len(network_rows) != protein_count or min(positions, default=-1) < 0 or len(set(positions)) != protein_count:
        _invalid("The CSV file needs exactly one network record per protein column, identified by id.")
    weights = np.zeros((protein_count, protein_count), dtype=np.float64)
    weights[positions, :] = _parse_numbers(
        [[cells[row][column] for column in protein_columns] for row in network_rows],
        [row_lines[row] for row in network_rows], protein_names, False,
    )
    if np.any(weights < 0):
        _invalid("Network weights must be nonnegative.")
    if not np.array_equal(weights, weights.T):
        _invalid("Network weights must form an exactly symmetric matrix.")

    if not sample_rows:
        _invalid("The CSV file contains no sample records.")
    sample_ids = [cells[row][id_column] for row in sample_rows]
    if any(identifier == "" for identifier in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        _invalid("Sample records need unique, nonempty ids.")
    roles = [cells[row][split_column] for row in sample_rows]
    is_train = np.array([role == "train" for role in roles], dtype=bool)
    is_test = np.array([role == "test" for role in roles], dtype=bool)
    if not np.all(is_train | is_test):
        row = sample_rows[int(np.flatnonzero(~(is_train | is_test))[0])]
        _invalid(f"Line {row_lines[row]}: split must be train or test.")
    if np.count_nonzero(is_train) < 2:
        _invalid("At least two train samples are required.")
    sample_lines = [row_lines[row] for row in sample_rows]
    features = _parse_numbers(
        [[cells[row][column] for column in protein_columns] for row in sample_rows], sample_lines, protein_names, False
    )
    label_values = []
    for column in label_columns:
        values = _parse_numbers(
            [[cells[row][column]] for row in sample_rows], sample_lines, [header[column]], True
        ).reshape(-1)
        train_values = values[is_train]
        test_values = values[is_test]
        if np.any(np.isnan(train_values)) or np.any((train_values != 0) & (train_values != 1)):
            _invalid(f"Column {header[column]} needs a 0/1 label for every train sample.")
        if np.all(np.isnan(test_values)):
            test_values = np.empty(0, dtype=np.float64)
        elif np.any(np.isnan(test_values)) or np.any((test_values != 0) & (test_values != 1)):
            _invalid(f"Column {header[column]} needs 0/1 labels for all test samples or for none.")
        label_values.append((train_values, test_values))
    test_rows = [row for row, flag in zip(sample_rows, is_test) if flag]
    train_rows = [row for row, flag in zip(sample_rows, is_train) if flag]
    _require_empty(cells, test_rows, fold_columns, row_lines, header)
    folds = _parse_numbers(
        [[cells[row][column] for column in fold_columns] for row in train_rows],
        [row_lines[row] for row in train_rows], [header[column] for column in fold_columns], False,
    ).reshape(len(train_rows), len(fold_columns))
    if np.any((folds < 1) | (folds != np.trunc(folds))):
        _invalid("Fold IDs in cv columns must be positive integers.")

    dataset = {
        "XData": features[is_train].T.copy(),
        "XTest": features[is_test].T.copy(),
        "YabtData": label_values[0][0], "YmtaData": label_values[1][0], "YwmhData": label_values[2][0],
        "YabtTest": label_values[0][1], "YmtaTest": label_values[1][1], "YwmhTest": label_values[2][1],
        "Wppi": weights,
        "Lppi": ppi_laplacian(weights),
        "IdxProtein": np.array(protein_names, dtype=object),
        "NumProtein": protein_count,
        "IdxCluster": cluster_index.astype(np.int64),
        "NumCluster": cluster_count,
        "CVindex": folds.T.copy(),
        "IdxSampleData": np.array([identifier for identifier, flag in zip(sample_ids, is_train) if flag], dtype=object),
        "IdxSampleTest": np.array([identifier for identifier, flag in zip(sample_ids, is_test) if flag], dtype=object),
    }
    return split_validset(dataset, require_cv=False)
