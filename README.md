---

# NeuroFANN: identification of neuropathological subtypes in dementia with plasma proteins by using functionally annotated neural network

---

## Publication
<b>Sunghong Park<sup>†</sup></b>, Doyoon Kim<sup>†</sup>, Ji-Hye Choi, Chang Hyung Hong, Sang Joon Son, Hyun Woong Roh, Hyunjung Shin<sup>&#42;</sup>, Hyun Goo Woo<sup>&#42;</sup>. <b>"NeuroFANN: identification of neuropathological subtypes in dementia with plasma proteins by using functionally annotated neural network"</b>. <span style="color:#3e8edc"><b><i> Briefings in Bioinformatics</i></b>, 26(4):bbaf336, August 2025.</span> [[Paper](https://doi.org/10.1093/bib/bbaf366)]

## Abstract
Dementia diagnosis relies on identifying neuropathological features, such as beta-amyloid (Aβ) deposition, medial temporal lobe atrophy (MTA), and white matter hyperintensity (WMH). Recently, plasma protein biomarkers have emerged as a cost-effective and less invasive tool for identifying neuropathological features, enhanced by machine learning (ML) for precise diagnosis. However, most ML studies fail to account for protein-protein interactions (PPIs) and synergetic effects between proteins, overlooking their collective contributions to disease mechanisms. Additionally, the lack of consideration for functional properties may result in the redundant and imbalanced representation of proteins and their functions, potentially limiting the effectiveness of dementia diagnosis. In this study, we propose NeuroFANN, a method designed to classify three neuropathological subtypes of dementia—positivity for Aβ, MTA, and WMH—using plasma protein biomarkers. A key feature of NeuroFANN is combining the PPI network-based synergetic effects with the functional annotation-based protein biomarker clustering. NeuroFANN extracts synergetic effects by propagating independent effects of proteins across the PPI network, which are then aggregated in functional protein clusters, enabling global PPI awareness and capturing the biological properties of protein biomarkers. From a South Korean cohort, 54 proteins were identified as plasma protein biomarkers for dementia subtypes and grouped into 16 clusters. NeuroFANN outperformed comparison methods in classifying dementia subtypes, with its core components validated as key contributors to superior performance. Additionally, the risk scores predicted by NeuroFANN showed a strong association with longitudinal cognitive decline, demonstrating its potential as a valuable diagnostic tool in clinical settings.

<b>Keywords</b>: Dementia; Neuropathological subtype; Plasma proteomic profile; Protein-protein interaction; Biologically functional annotation; Graph-based machine learning.

![Image](https://github.com/user-attachments/assets/84bfe8d7-1e8b-44b1-b100-35b05ca35932)

---

## Data description

### Study participants
We recruited participants from the Biobank Innovations for chronic Cerebrovascular disease With ALZheimer’s disease Study (BICWALZS) at Ajou University Hospital (Suwon, Republic of Korea) [[Paper](https://doi.org/10.30773/pi.2021.0335)]. The neuropathological subtypes of participants were determined by clinicians based on neuroimaging, where Aβ-positivity was determined by 18F-flutemetamol PET, and positivity for MTA and WMH was determined by 3.0 Tesla MRI. We divided the participants into the discovery and validation cohorts, where the validation cohort included participants with 2-year follow-up on cognitive assessments, and the participants with only baseline information were included in the discovery cohort. The demographic and clinical characteristics of study participants are presented in Table below.

|      Characteristics     | Total   participants (N = 475) | Discovery cohort   (N = 348) | Validation cohort   (N = 127) | P-value |
|:------------------------:|:------------------------------:|:----------------------------:|:-----------------------------:|:-------:|
|   Age, median (IQR), yr  |           73 (67–77)           |          72 (67–77)          |           73 (67–78)          |  0.6343 |
|      Female, No. (%)     |           332 (69.9)           |          241 (69.3)          |           91 (71.7)           |  0.6146 |
|    MMSE, median (IQR)    |           24 (20–27)           |          24 (20–27)          |           24 (22–27)          |  0.0525 |
|   CDR-SB, median (IQR)   |          2.5 (1.5–4.0)         |         2.5 (1.5–4.5)        |         2.0 (1.5–3.0)         |  0.0921 |
|     GDS, median (IQR)    |             3 (3–4)            |            3 (3–4)           |            3 (3–4)            |  0.2841 |
| APOE ε2 carrier, No. (%) |            68 (14.3)           |           48 (13.8)          |           20 (15.7)           |  0.5912 |
| APOE ε4 carrier, No. (%) |           129 (27.2)           |           91 (26.1)          |           38 (29.9)           |  0.4144 |
|  Aβ-positivity, No. (%)  |           142 (29.9)           |          100 (28.7)          |           42 (33.1)           |  0.3621 |
|  MTA-positivity, No. (%) |           106 (22.3)           |           81 (23.3)          |           25 (19.7)           |  0.4065 |
|  WMH-positivity, No. (%) |           172 (36.2)           |          121 (34.8)          |           51 (40.2)           |  0.2806 |

### Plasma samples
The plasma samples were profiled by the Olink Target 96 Neurology and Olink Target 48 Cytokine panels. The former comprises 92 established assays associated with neurobiological diseases, while the latter includes 45 selected assays highly relevant to inflammatory processes, more details can be found at: https://olink.com, and the assayed proteins are listed in Supplementary Table S1. The raw data for protein expression underwent quality control (QC) through both internal and external controls in the panels. We excluded proteins with missing frequency over 10% and imputed missing values by using the k-nearest neighbor method. Subsequently, the protein expression values were standardized by using Z-score normalization and then transformed by the logistic function for scaling.

---

## Model implementation

### Overall process
This study begins with the identification of plasma proteins associated with neuropathological subtypes of dementia through differential expression analysis (DEA), followed by the clustering of these protein biomarkers based on their functional annotations. Next, NeuroFANN is used to predict individual risks for dementia subtypes. The model projects the independent effects of proteins that passed the DEA criteria onto the PPI network, capturing synergetic effects that reflect the global properties of PPIs. The synergetic effects are then aggregated within predefined clusters, reflecting the functional annotations of plasma protein biomarkers.

<b>Differential expression analysis</b>
- The method identifies differentially expressed proteins (DEPs) by comparing plasma protein expression between neuropathological subtype-positive and subtype-negative groups using the ‘limma’ R/Bioconductor package. The union of DEPs across all subtypes serves as candidate plasma biomarkers.

<b>Biologically informed protein clustering</b>
- This approach clusters biomarkers based on biological functionality. It involves first identifying Gene Ontology (GO) groups significantly associated with biomarkers via DAVID's Functional Annotation Clustering. Subsequently, hierarchical clustering groups biomarkers according to shared functional annotations using the Jaccard distance metric and Ward’s linkage method.

<b>Functionally annotated neural network</b>
- NeuroFANN captures biomarker interactions and functional importance through three steps: (1) network propagation of independent biomarker effects via a PPI network, (2) aggregation of propagated signals at the cluster level using trainable, protein-specific importance weights, and (3) prediction of individual risks for neuropathological subtypes using logistic regression on aggregated cluster signals.

### Code description

<b>MATLAB (version R2024b)</b>
- <b>Software</b>
  - MATLAB (version R2024b)

- <b>Main code</b>
  - NeuroFANN
  
- <b>Data setting</b>
  - Xdata: independent effect of target proteins
  - Ydata: real diagnosis label set for biomarkers
  - Split two data into train/valid/test sets

- <b>Parameter setting</b>
  - Uprot: propagation parameter
  - Aclus: importance parameter
  - Babt, Bmta, Bwmh: prediction parameters

- <b>Model parameter</b>
  - MaxEpoch: maximum number of epoch
  - LearnRate: learning rate
  - RegCoeff: regularization coefficient

---
