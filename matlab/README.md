# NeuroFANN for MATLAB and GNU Octave

This folder trains and evaluates NeuroFANN (functionally annotated neural network) on `../dataset/sample.csv` or on
any dataset in the same format. The same source runs unchanged in MATLAB and in GNU Octave. Its results agree with
the earlier MATLAB R2025a release and with the Python version in `../python` (see Verification).

## Quick start

```matlab
cd matlab
[ScoreABT, ScoreMTA, ScoreWMH, Results] = main_NeuroFANN();
```

The default run uses all 100 predefined split rows × 5 folds × 500 epochs (500 models); it takes about 7 minutes in
MATLAB R2025a (measured for the previous release) and 35 minutes in GNU Octave 8.4. For the paper's protocol
(20 split iterations, 100 models) or a quick check:

```matlab
[ScoreABT, ScoreMTA, ScoreWMH, Results] = main_NeuroFANN([], struct('NumIter', 20, 'Verbose', true));
[ScoreABT, ScoreMTA, ScoreWMH, Results] = main_NeuroFANN([], struct('NumIter', 1, 'MaxEpoch', 20));
files = data_writeresults(Results, 'neurofann_results');
```

From a shell: `octave-cli --eval "cd matlab; main_NeuroFANN([], struct('NumIter', 1, 'Verbose', true));"`.
`main_NeuroFANN` adds its own folder to the path for the duration of the call and restores the path afterwards.

## Inputs

`main_NeuroFANN(dataSource, overrides)`

| `dataSource` | Meaning |
|---|---|
| `[]` or omitted | `../dataset/sample.csv` |
| `'file.csv'` | a dataset file in the format of `../dataset/README.md` |
| `'file.mat'` | a MAT file with a scalar struct variable `Dataset` |
| struct | a dataset struct, for example from `data_readcsv` |

| Option | Default | Meaning |
|---|---|---|
| `NumIter` | rows of `CVindex` (100) | split iterations used |
| `NumFold` | largest fold ID of the first row (5) | folds per iteration; every used row must contain exactly the IDs 1…NumFold |
| `MaxEpoch` | 500 | candidate epochs per fold, including the initial weights |
| `LearnRate` | 0.001 | Adam step size |
| `RegCoeff` | 0.005 | coefficient of the squared L2 penalty on all parameters |
| `Seed` | 1 | iteration `i` initializes its heads with seed `Seed + i − 1` |
| `StoreHistory` | false | keep the weights of every epoch |
| `Verbose` | false | print one line per fold and an AUROC summary |

Dataset fields: `XData` (proteins × discovery samples), `XTest` (proteins × test samples), `YabtData`, `YmtaData`,
`YwmhData` (0/1), optional `YabtTest`, `YmtaTest`, `YwmhTest`, `Wppi` and/or `Lppi`, `IdxCluster` (1…NumCluster per
protein), `NumProtein`, `NumCluster`, `CVindex` (iterations × discovery samples), optional `IdxProtein`,
`IdxSampleData`, `IdxSampleTest`. `split_validset` checks and normalizes all of them and reports problems with
`NeuroFANN:*` error identifiers.

## Outputs

- `ScoreABT`, `ScoreMTA`, `ScoreWMH`: `NumIter × NumFold` cell arrays; each cell is the 1 × NumTest risk vector of
  one model.
- `Results.OutOfFoldABT/MTA/WMH`: `NumIter × NumDiscovery` validation-fold risks.
- `Results.Folds{i, f}`: selected epoch, its validation loss, weights, Adam state, loss histories, sample indices,
  seed and training time (plus `WeightEpoch` with `StoreHistory`).
- `Results.Evaluation` (`model_evaluate`): AUROC of every model on the test samples (`TestAUROC`, I × F × 3), the
  AUROC of the ensemble mean risk (`EnsembleTestAUROC`), out-of-fold AUROC per iteration, and mean, SD, minimum and
  maximum of each. AUROC is the Mann–Whitney statistic with average ranks for ties; it is NaN when a target has a
  single class.
- `Results.Parameter`, `DataSource`, `Runtime`, `ElapsedSeconds`.

`data_writeresults(Results, folder)` writes `predictions_test.csv`, `predictions_oof.csv`, `auroc_models.csv`,
`auroc_summary.csv` and `run_summary.json` (numbers with 17 significant digits, empty cells for undefined values).

## Functions

| File | Role |
|---|---|
| `main_NeuroFANN` | entry point: load, normalize, train all folds, evaluate |
| `data_readcsv`, `data_writecsv` | strict CSV reader and byte-exact writer |
| `split_validset` | dataset validation and normalization; derives `Lppi` from `Wppi` |
| `ppi_laplacian` | normalized Laplacian `I − D^(−1/2) W D^(−1/2)` in a fixed evaluation order |
| `split_cvindex` | reproducible label-stratified fold assignments for new data |
| `model_options` | option defaults and validation |
| `model_initialize` | data of one iteration and fold |
| `param_initialize` | initial weights: `Uprot = 1`, `Aclus = 0`, heads uniform in ±sqrt(6/(K+1)) |
| `rand_mt19937` | MT19937 stream identical to MATLAB `RandStream('mt19937ar')` |
| `param_reshape_` | unpack the 156-parameter vector; cluster softmax |
| `factor_prop`, `solver_prop` | LU factorization of `diag(u) + L` with a condition check; forward and transpose solves |
| `model_forward`, `loss_measure`, `model_backward` | propagation, pooling and logistic heads; loss; exact gradient |
| `init_adamopt`, `param_update` | Adam state and update |
| `param_training` | epoch loop with validation-based selection |
| `metric_auroc`, `model_evaluate`, `data_writeresults` | evaluation and export |

## Model and training

For proteins with propagation weights `u` and normalized Laplacian `L`, hidden signals are
`H = (diag(u) + L)^(−1) diag(u) X`. Cluster `c` pools its members with softmax attention, `Z_c = Σ_i a_i H_i`, and
three logistic heads give the Aβ, MTA and WMH risks. The objective is the sum of the three mean binary cross-entropies
plus `RegCoeff · ‖w‖²`. Each fold evaluates `MaxEpoch` candidate weight vectors (`MaxEpoch − 1` Adam updates), keeps
the earliest candidate with the lowest validation loss, restores the Adam state of that candidate, and predicts the
test samples with it.

The propagation gradient is the exact adjoint `∂J/∂u = rowsum(M^(−T) G ∘ (X − H)) + 2·RegCoeff·u` with
`M = diag(u) + L`. The propagation-weight gradient printed in the paper and the one in the original GitHub code are
not derivatives of the loss; the finite-difference tests reject both
(`test_numerics/publishedAndOriginalPropagationGradientsAreInexact`). Losses and probabilities use overflow-free
forms, and a singular or nearly singular `diag(u) + L` stops training with `NeuroFANN:SingularPropagation` instead of
returning an inaccurate solution.

## Changes from the previous MATLAB code

- Reads the CSV dataset by default and derives the Laplacian from the network, reproducing the stored `Lppi` bit for
  bit; MAT files and structs still work.
- Runs in GNU Octave as well as MATLAB: `RandStream` and `decomposition`, which Octave lacks, were replaced by
  `rand_mt19937` and `factor_prop`/`solver_prop`, and `validateattributes`, whose error identifiers differ between
  the two runtimes, by explicit checks. The generator reproduces MATLAB's `mt19937ar` stream exactly, so seeded
  initial weights are unchanged.
- The propagation factorization now checks the reciprocal condition number (previously disabled).
- New evaluation (`metric_auroc`, `model_evaluate`), result export (`data_writeresults`), fold generation
  (`split_cvindex`) and CSV writing; all failures carry `NeuroFANN:*` identifiers that are identical in both runtimes.
- Training mathematics, parameter packing, seeds and the epoch/selection protocol are unchanged.

## Tests

```matlab
cd matlab/tests
summary = run_tests();
summary = run_tests('test_numerics.m');
```

`octave-cli --eval "cd matlab/tests; run_tests"` exits with status 1 if a test fails. The 81 tests check the forward
pass against a scalar elimination oracle, every gradient entry against central differences, the published gradient
formula, numerical failure modes, input contracts (with the same error identifiers as Python), Adam in closed form,
the full training protocol, the generator against the MT19937 reference outputs and MATLAB R2025a initial weights,
the CSV format (24 malformed files, byte-exact round trips), stratified fold generation, AUROC against the pairwise
definition, the result files, and complete runs against MATLAB R2025a outputs of the previously validated release
(`tests/fixtures/matlab_r2025a_reference.json`).

## Verification

| Check | Result |
|---|---|
| Test suite, MATLAB R2025a on Windows | 81 passed in 54 s, including `rand_mt19937` against `RandStream('mt19937ar')` |
| Test suite, GNU Octave 8.4.0 on Linux | 80 passed, 1 skipped (`RandStream` exists only in MATLAB), 88 s |
| 1 iteration × 5 folds × 20 epochs vs MATLAB R2025a release (Octave) | same selected epochs; risks ≤ 2.2e-16, weights ≤ 3.7e-16, validation losses ≤ 3.6e-15 |
| 1 × 5 × 500 vs MATLAB R2025a release (Octave) | same selected epochs; risks ≤ 1.7e-16, weights ≤ 4.4e-16 |
| 100 × 5 × 500 full default run vs MATLAB R2025a release (Octave, 35 min) | all 500 selected epochs, all 1,500 per-model test AUROCs, all 300 out-of-fold AUROCs and the ensemble AUROCs identical; risks ≤ 2.5e-13 and weights ≤ 3.9e-11 in 499 of 500 models |

In the full run, the remaining model (iteration 88, fold 1) differs by up to 1.6e-9 in risk and 2.6e-7 in weights:
there, rounding-level gradient differences are amplified by Adam over several hundred updates, as diagnosed for the
previous port. The full-run reference is `validation/sample_100iter_5fold_500epoch.mat` of the previous repository
(not copied here). The AUROC tables written by this run (`auroc_models.csv`, `auroc_summary.csv`) are byte-identical
to those of the Python full run, and the averaged risks in the prediction tables differ by less than 1.4e-11.

The MATLAB run used a folder path with spaces, brackets and Korean characters. In MATLAB the 20- and 500-epoch
reference comparisons run inside the test suite with tolerances fixed before running (1e-10; 1e-8 for 500-epoch
risks; 1e-6 for 500-epoch weights); the full 100-split run was executed in GNU Octave and Python. Results on
`sample.csv` are near chance level because its features are uninformative (`../dataset/README.md`).
