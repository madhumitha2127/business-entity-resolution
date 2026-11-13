
<div align="center">

# 🔍 BUSINESS ENTITY RESOLUTION

### Intelligent Matching · Data Processing · Machine Learning

<img src="https://img.shields.io/badge/PYTHON-0B0F14?style=for-the-badge&logo=python&logoColor=3DF2E0" />
<img src="https://img.shields.io/badge/LIGHTGBM-0B0F14?style=for-the-badge&logo=lightgbm&logoColor=3DF2E0" />
<img src="https://img.shields.io/badge/STATUS-COMPLETED-0B0F14?style=for-the-badge&labelColor=0B0F14&color=3DF2E0" />

</div>

---

## 🖥️ `01 / OVERVIEW`

Business Entity Resolution identifies records that refer to the same real-world business across multiple data sources.

This solution uses text normalization, candidate generation, feature engineering and machine learning to identify matching business entities.

## ⚙️ `02 / PROCESSING_PIPELINE`

```text
DATA INPUT
    |
    v
DATASET INSPECTION
    |
    v
TEXT NORMALIZATION
    |
    v
BLOCKING & CANDIDATE GENERATION
    |
    v
TRAINING PAIR CREATION
    |
    v
FEATURE ENGINEERING
    |
    v
LIGHTGBM MODEL TRAINING
    |
    v
VALIDATION & THRESHOLD SELECTION
    |
    v
TEST PREDICTION
    |
    v
OUTPUT VALIDATION
```

## 🧠 `03 / MODEL_DETAILS`

| Component | Technology |
|---|---|
| Programming | Python |
| ML Model | LightGBM 4.7.0 |
| Feature processing | Feature Engineering |
| Prediction threshold | 0.81 |
| License | MIT |

## 📊 `04 / VALIDATION_METRICS`

<div align="center">

| Metric | Result |
|:---:|:---:|
| F0.5 Score | **0.9811** |
| Precision | **0.9894** |
| Recall | **0.9491** |
| AUC | **0.99936** |

</div>

<sub>These are validation-set results, not hidden test-set results.</sub>

## 📁 `05 / OUTPUT_DETAILS`

The final output includes:

- `matching_results.tsv` – predicted matching IDs for each Source 1 entity.
- `candidate_pairs.tsv` – candidate IDs supplied to the matching stage.

**Dataset coverage:** 1,732,544 Source 1 entities.

The official challenge validator passed with ID checking enabled. Additional checks confirmed no duplicate Source 1 rows or IDs and verified that predicted IDs existed in the test data.

## 🚀 `06 / SETUP`

Clone the repository:

```bash
git clone https://github.com/madhumitha2127/business-entity-resolution.git
```

Navigate into the directory:

```bash
cd business-entity-resolution
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Source code is available in the `src` directory.

## 🔬 `07 / KEY_CONCEPTS`

- Entity matching
- Text normalization
- Blocking and candidate generation
- Feature engineering
- Gradient boosting
- Model validation
- Threshold selection

---

<div align="center">

### ⚡ MATCHING DATA. FINDING CONNECTIONS.

**Built with Python and machine learning.**

</div>
