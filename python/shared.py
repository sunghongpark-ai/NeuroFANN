import math
from pathlib import Path

import numpy as np

MAXIMUM_SEED = 4294967295
RESERVED_COLUMNS = ("record", "id", "split", "y_abt", "y_mta", "y_wmh")
TARGETS = ("abt", "mta", "wmh")


class NeuroFANNError(ValueError):
    def __init__(self, identifier, message):
        self.identifier = identifier
        super().__init__(f"{identifier}: {message}")


def default_dataset_path():
    return Path(__file__).resolve().parents[1] / "dataset" / "sample.csv"


def finite_scalar(value, name, identifier="NeuroFANN:InvalidParameter", allow_bool=False):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exception:
        raise NeuroFANNError(identifier, f"{name} must be a finite real scalar.") from exception
    allowed = "biuf" if allow_bool else "iuf"
    if array.size != 1 or array.dtype.kind not in allowed:
        raise NeuroFANNError(identifier, f"{name} must be a finite real scalar.")
    number = float(array.reshape(-1)[0])
    if not np.isfinite(number):
        raise NeuroFANNError(identifier, f"{name} must be a finite real scalar.")
    return number


def integer_scalar(value, name, lower, upper=np.inf, identifier="NeuroFANN:InvalidParameter"):
    number = finite_scalar(value, name, identifier)
    if not number.is_integer() or number < lower or number > upper:
        raise NeuroFANNError(identifier, f"{name} must be an integer from {lower:g} to {upper:g}.")
    return int(number)


def logical_flag(value, name, identifier="NeuroFANN:InvalidParameter"):
    number = finite_scalar(value, name, identifier, allow_bool=True)
    if number not in (0.0, 1.0):
        raise NeuroFANNError(identifier, f"{name} must be a scalar logical flag.")
    return bool(number)


def text_path(value, identifier="NeuroFANN:InvalidDataset"):
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value:
        return Path(value)
    raise NeuroFANNError(identifier, "Provide the file path as text.")


def format_number(value):
    number = float(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "Inf" if number > 0 else "-Inf"
    return format(number, ".17g")


def is_csv_token(text):
    return (
        isinstance(text, str)
        and len(text) > 0
        and all(32 <= ord(character) <= 126 for character in text)
        and "," not in text
        and '"' not in text
        and not text[0].isspace()
        and not text[-1].isspace()
    )
