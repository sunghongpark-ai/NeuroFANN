# NeuroFANN sample dataset

`sample.csv` is the input of the MATLAB (`../matlab`) and Python (`../python`) implementations. It holds the complete
sample that ships with the original NeuroFANN code (`CODE/Sample.mat`): protein features, the three binary
neuropathology labels, the protein–protein interaction (PPI) network, the functional protein clusters and the 100
predefined cross-validation splits, in one plain-text file.

| Property | Value |
|---|---|
| Size | 621,816 bytes, 531 lines, 160 columns |
| SHA-256 | `d1cf811aec81832af28031840bcff324188446b43ce396096ba7450e05d52d95` |
| Encoding | printable ASCII, LF line endings, no quoting |
| Proteins / clusters | 54 / 16 |
| Samples | 348 discovery (`train`), 127 validation (`test`) |
| Splits | `cv001` … `cv100`: 100 predefined 5-fold assignments of the discovery samples |

## Layout

Every line starts with a record type. The header is

```
record,id,split,y_abt,y_mta,y_wmh,cv001,...,cv100,ACVRL1,...,VWC2
```

| Record | Rows | `id` | `split` | `y_*` | `cv*` | Protein columns |
|---|---:|---|---|---|---|---|
| `cluster` | 1 | `cluster` | empty | empty | empty | cluster ID (1–16) of each protein |
| `network` | 54 | protein name | empty | empty | empty | PPI weights of that protein (row of the 54 × 54 matrix W) |
| `sample` | 348 | `D001` … `D348` | `train` | 0/1 | fold ID 1–5 | feature values |
| `sample` | 127 | `V001` … `V127` | `test` | 0/1 | empty | feature values |

`y_abt`, `y_mta` and `y_wmh` are amyloid-β positivity, medial temporal lobe atrophy and white matter
hyperintensity. Test labels may be left empty for all test samples of a target (prediction without evaluation).

## Exactness

- Every number is written with `%.17g`, so each IEEE-754 double is recovered bit for bit. All values equal the
  original `Sample.mat` exactly, and the MATLAB/Octave writer (`data_writecsv.m`) and the Python writer
  (`data_writecsv.py`) both regenerate this file byte for byte.
- The normalized Laplacian is not stored. Both readers rebuild it from W as `L = I − D^(−1/2) W D^(−1/2)`
  (`ppi_laplacian`): degrees are left-to-right row sums, `1/sqrt(degree)` is 0 for the 24 isolated proteins, and
  `L(i,j) = δij − (s(i)·W(i,j))·s(j)`. This evaluation order reproduces the original `Lppi` bit for bit, including
  its 22 entries that differ from their transpose by one rounding (at most 1.1e-16).
- The readers reject anything ambiguous with an error naming the line and column: quotes, non-ASCII bytes, missing or
  duplicate columns, gaps in `cv` numbering, non-decimal or overflowing numbers, asymmetric or negative weights,
  unused cluster IDs, duplicate sample ids, labels other than 0/1, and fold IDs on test rows. A UTF-8 byte-order
  mark, CRLF line endings, blank lines and spaces around fields are accepted.

## Contents

| | Discovery (train) | Validation (test) |
|---|---:|---:|
| Samples | 348 | 127 |
| Aβ-positive | 100 (28.7%) | 42 (33.1%) |
| MTA-positive | 81 (23.3%) | 25 (19.7%) |
| WMH-positive | 121 (34.8%) | 51 (40.2%) |

These counts are those of the study cohorts in the paper (Table 1). The PPI network has 51 weighted edges (weights
0.165–0.832) and 24 proteins without interactions. Fold sizes are 67, 73, 71, 70 and 67 in every split row.

The cluster IDs in the file are numbered differently from Supplementary Table S3 of the paper; the protein members
match the table exactly:

| ID in file | Table S3 No. | Biomarker cluster | Proteins |
|---:|---:|---|---|
| 1 | 2 | Cytokine associated signal transduction | CD300LF, EDA2R |
| 2 | 10 | Neuron development associated cell motility | FLRT2, NRP2, PLXNB3, UNC5C |
| 3 | 7 | Macromolecule associated signal transduction | ACVRL1, CDH3, MMP12, SMOC2 |
| 4 | 8 | Macromolecule associated signaling receptor | FCRL2, RGMB |
| 5 | 12 | Positive regulation of adaptive immune response | PVR, ULBP2 |
| 6 | 4 | Extracellular matrix | BCAN, VWC2 |
| 7 | 13 | Positive regulation of metabolic process | GFRA1, LAT, NTRK3, RSPO1, SMPD1 |
| 8 | 9 | Molecular transducer activity | CD300C, CLEC10A, NCAN, SCARB2 |
| 9 | 1 | Chemokines | CCL11, CCL13, CXCL9 |
| 10 | 5 | Focal adhesion | LAYN, SCARF2 |
| 11 | 14 | Programmed cell death | ROBO2, TNFRSF12A |
| 12 | 11 | Neuron development associated signal transduction | CNTN5, EFNA4, EPHB6, TNFRSF21 |
| 13 | 6 | Leukocyte activation | CTSC, IL27, TNFSF12 |
| 14 | 16 | Signal transduction | ADAM22, ADAM23, DKK4, IL17C |
| 15 | 15 | Scavenger receptor type A | MSR1, SCARA5 |
| 16 | 3 | Cytokines | CCL19, CCL3, CSF1, IL12A, IL15, IL18, IL1B, IL33, TNF |

## What results to expect

The feature values in this sample behave like independent Uniform(0, 1) noise and carry no detectable label signal:

- mean 0.498 and SD 0.2895 (uniform SD 0.2887); pooled Kolmogorov–Smirnov test against U(0, 1): D = 0.0053,
  p = 0.46 (n = 25,650); 2 of 54 proteins have per-protein KS p < 0.05;
- mean absolute correlation between proteins 0.037, the value expected for independent columns;
- Mann–Whitney tests of each protein against each label: 2–3 of 54 nominal p < 0.05 per label, none significant after
  Benjamini–Hochberg correction.

The paper describes features as z-scored and logistic-transformed cohort measurements, which would not be uniform.
The sample therefore exercises the full pipeline but cannot reproduce the published performance (average AUROC
0.832, Supplementary Table S4). A complete run (100 split rows × 5 folds × 500 epochs) gives near-chance test AUROC:
mean over 500 models 0.546 (Aβ), 0.501 (MTA), 0.510 (WMH); ensemble risk 0.565, 0.497, 0.515. The paper used 20 split
iterations (100 models); set `NumIter` to 20 for that protocol.

## Using your own data

Write a file with the same layout, or build a dataset struct/dictionary and save it with `data_writecsv`. Protein
names must be plain CSV fields and must not be `record`, `id`, `split`, `y_abt`, `y_mta`, `y_wmh` or `cv<number>`.
If you have no predefined splits, read the file and create stratified folds with `split_cvindex` (stratified by the
joint label pattern, reproducible from a seed):

```matlab
dataset = data_readcsv('my_data.csv');
dataset.CVindex = split_cvindex(dataset, 20, 5, 1);
data_writecsv(dataset, 'my_data_with_folds.csv');
```

```python
from python import data_readcsv, data_writecsv, split_cvindex
dataset = data_readcsv("my_data.csv")
dataset["CVindex"] = split_cvindex(dataset, 20, 5, 1)
data_writecsv(dataset, "my_data_with_folds.csv")
```

Keep the file byte-exact under Git: `.gitattributes` in this folder disables line-ending conversion for CSV files.
