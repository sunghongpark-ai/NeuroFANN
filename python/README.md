# NeuroFANN for Python

A NumPy/SciPy translation of the MATLAB implementation in `../matlab`, file for file and function for function. It
reads `../dataset/sample.csv`, trains NeuroFANN with the same formulas, operation sequence, seeds and
epoch-selection protocol, accepts and rejects the same inputs with the same error identifiers, and reproduces the
MATLAB results (see Verification).

## Install

Python 3.10 or newer:

```bash
python -m pip install -r python/requirements.txt
python -m pip install -r python/requirements-dev.txt
```

`requirements.txt` holds NumPy and SciPy; `requirements-dev.txt` adds pytest for the tests.

## Run

From the repository root:

```bash
python python/main_NeuroFANN.py --iterations 1 --epochs 20 --verbose
python python/main_NeuroFANN.py --iterations 20 --output-dir results --output results/folds.npz
python -m python.main_NeuroFANN --help
```

Without options the run uses all 100 split rows × 5 folds × 500 epochs (500 models, about 11 minutes on one CPU core);
`--iterations 20` follows the paper (100 models). The command writes `predictions_test.csv`, `predictions_oof.csv`,
`auroc_models.csv`, `auroc_summary.csv` and `run_summary.json` to `--output-dir` (default `neurofann_results`), prints
a JSON summary, and exits with status 1 and a `NeuroFANN:*` message on invalid input. Runs are deterministic; to
rule out any dependence on BLAS threading, set `OPENBLAS_NUM_THREADS=1` (or `MKL_NUM_THREADS=1`) before starting
Python. The library never changes thread settings itself.

| Option | MATLAB option | Default |
|---|---|---|
| `--data` | `dataSource` | `dataset/sample.csv` (also `.mat` with a `Dataset` struct, or `.npz`) |
| `--iterations` | `NumIter` | every `CVindex` row |
| `--folds` | `NumFold` | largest fold ID |
| `--epochs` | `MaxEpoch` | 500 |
| `--learning-rate` | `LearnRate` | 0.001 |
| `--regularization` | `RegCoeff` | 0.005 |
| `--seed` | `Seed` | 1 |
| `--store-history` | `StoreHistory` | off |
| `--verbose` | `Verbose` | off |
| `--output` | | optional compressed NPZ with every fold state (readable with `allow_pickle=False`) |

## API

```python
from python import main_NeuroFANN, data_writeresults, data_readcsv, split_cvindex

score_abt, score_mta, score_wmh, results = main_NeuroFANN(None, {"NumIter": 1, "MaxEpoch": 20})
print(results["Evaluation"]["TestAUROCMean"])
data_writeresults(results, "results")
```

`main_NeuroFANN(data_source=None, overrides=None)` accepts `None` (the sample), a path to a `.csv`, `.mat` or `.npz`
file, or a dataset dictionary with the MATLAB field names; `overrides` uses the MATLAB option names. It returns three
score arrays of shape `(NumIter, NumFold, NumTest)` and a `results` dictionary with `OutOfFoldABT/MTA/WMH`
(`(NumIter, NumDiscovery)`), `Folds[iteration][fold]` (selected epoch, weights, Adam state, loss histories, sample
indices, seed), `Evaluation` (per-model, ensemble and out-of-fold AUROC with summary statistics), `Parameter`,
`DataSource`, `Runtime` and `ElapsedSeconds`. `save_results(path, *outputs)` stores everything as a compressed NPZ.
Errors are `NeuroFANNError` (a `ValueError`) with the same `identifier` as in MATLAB, for example
`NeuroFANN:InvalidCSV`.

| Python | MATLAB |
|---|---|
| `main_NeuroFANN.py` … `split_validset.py` | the `.m` file of the same name; every function keeps its name, arguments and checks (`shared.py` holds the common checks) |
| score arrays `(I, F, T)` | `I × F` cell arrays of `1 × T` vectors |
| `IdxTrain`, `IdxValid`, `ClusterMembers` | zero-based (one-based in MATLAB); fold IDs, cluster IDs and epochs stay one-based |
| one-dimensional `float64` arrays | column vectors (row vectors for labels) |
| new dictionary returned by each step | struct returned by each step |
| `rand_mt19937` via a private `RandomState` | `rand_mt19937.m`; neither touches the global generator |
| LAPACK `dgetrf`/`dgecon`/`dgetrs` | `lu(..., 'vector')`, `rcond` and triangular solves |

## Tests

```bash
cd python
python -m pytest
```

244 tests: forward pass against a scalar elimination oracle; every gradient entry against central differences on
symmetric, nonsymmetric, sparse and sample-sized problems; rejection of the published and original gradient formulas;
overflow, singularity and invalid-input behaviour with the MATLAB error identifiers; Adam in closed form; the training protocol with real and mocked
components; the generator against an independent MT19937 implementation, the MT19937 reference outputs and MATLAB
R2025a initial weights; the CSV format (24 malformed files, byte-exact round trips, CRLF/BOM tolerance); label-stratified
fold generation; AUROC against the pairwise definition and SciPy's Mann–Whitney U; result files; the command line;
comment- and docstring-free sources; and complete runs against MATLAB R2025a outputs
(`tests/fixtures/matlab_r2025a_reference.json`, absolute tolerances fixed before running: 1e-10 for 20 epochs, 1e-8
for 500-epoch predictions and 1e-6 for 500-epoch weights; selected epochs must be identical).

## Verification

244 tests passed with Python 3.13.16, NumPy 2.5.3 and SciPy 1.18.1 on Linux (17 s), with the oldest supported
versions, Python 3.10.20, NumPy 2.0.2 and SciPy 1.14.1 (18 s), and with Anaconda Python 3.13.9 on Windows in a folder
path with spaces, brackets and Korean characters (22 s).

| Comparison with MATLAB R2025a | Result |
|---|---|
| 1 iteration × 5 folds × 20 epochs | same selected epochs; risks ≤ 2.2e-16, weights ≤ 1.0e-15, validation losses ≤ 8.9e-16 |
| 1 × 5 × 500 | same selected epochs; risks ≤ 1.7e-16, weights ≤ 4.4e-16 |
| 100 × 5 × 500 (full default run, 11 min) | all 500 selected epochs, all 1,500 per-model test AUROCs, all 300 out-of-fold AUROCs and the ensemble AUROCs identical; test risks ≤ 2.4e-10, out-of-fold risks ≤ 2.6e-10; in 499 of 500 models risks ≤ 2.4e-12 and weights ≤ 3.8e-10 |

Bitwise equality with MATLAB is not expected: MATLAB and NumPy sum vectors in different orders and use different
`exp` and BLAS kernels, so single steps differ in the last bit and the table shows how far such differences
propagate. The full-run reference is `validation/sample_100iter_5fold_500epoch.mat` of the previous repository
(not copied here).
Weights differ by at most 4.2e-8, in one fold (iteration 88, fold 1), where rounding-level gradient differences are
amplified by Adam over several hundred updates; the previous port showed the same fold with roughly eight times larger
differences (risks 2.2e-9, weights 3.6e-7). On the sample the test AUROC is near chance level (mean 0.546, 0.501 and
0.510 for Aβ, MTA and WMH) because the sample's features are uninformative (`../dataset/README.md`).
