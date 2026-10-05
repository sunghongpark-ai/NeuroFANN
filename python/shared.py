from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.io import loadmat


class NeuroFANNError(ValueError):
    def __init__(self, identifier, message):
        self.identifier = identifier
        super().__init__(f"{identifier}: {message}")


def _protein_names(value):
    if type(value).__name__ == "MatlabOpaque":
        raise NeuroFANNError(
            "NeuroFANN:UnsupportedMetadata",
            "Save Dataset.IdxProtein as cellstr and use MAT v7 format for Python.",
        )
    array = np.asarray(value)
    if array.dtype.kind != "O":
        return array
    names = []
    for item in array.ravel(order="F"):
        text = np.asarray(item)
        if text.dtype.kind not in "US":
            raise NeuroFANNError("NeuroFANN:InvalidDataset", "Protein identifiers must be text or numeric values.")
        names.append("".join(text.ravel().tolist()))
    return np.asarray(names, dtype=str)


def load_dataset(data_source=None):
    if isinstance(data_source, Mapping):
        return dict(data_source)
    if data_source is None:
        data_source = Path(__file__).with_name("data_sample.mat")
    if not isinstance(data_source, (str, Path)):
        raise NeuroFANNError("NeuroFANN:InvalidDataset", "Provide a Dataset mapping or MAT/NPZ file path.")
    source = Path(data_source).expanduser()
    if not source.is_file():
        raise NeuroFANNError("NeuroFANN:MissingDataFile", f"Dataset file does not exist: {source}")
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}
    try:
        payload = loadmat(
            source, variable_names=["Dataset"], struct_as_record=False,
            squeeze_me=False, chars_as_strings=True,
        )
    except NotImplementedError as exception:
        raise NeuroFANNError("NeuroFANN:UnsupportedDataFile", "Use MAT v7 or NPZ; MATLAB v7.3 is not supported.") from exception
    except (ValueError, OSError, TypeError) as exception:
        raise NeuroFANNError("NeuroFANN:InvalidDataset", f"Cannot read dataset file: {exception}") from exception
    if "Dataset" not in payload:
        raise NeuroFANNError("NeuroFANN:InvalidDataset", "The MAT file must contain a variable named Dataset.")
    structure = payload["Dataset"]
    if not isinstance(structure, np.ndarray) or structure.size != 1:
        raise NeuroFANNError("NeuroFANN:InvalidDataset", "Dataset must be a scalar MATLAB struct.")
    structure = structure.item()
    if not hasattr(structure, "_fieldnames"):
        raise NeuroFANNError("NeuroFANN:InvalidDataset", "Dataset must be a scalar MATLAB struct.")
    dataset = {name: getattr(structure, name) for name in structure._fieldnames}
    if "IdxProtein" in dataset:
        dataset["IdxProtein"] = _protein_names(dataset["IdxProtein"])
    return dataset

