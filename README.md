
# 🟡 Business Entity Resolution

<p align="center">
  <img src="https://img.shields.io/badge/DOMAIN-Financial%20Data%20Intelligence-FF9F1C?style=for-the-badge" />
  <img src="https://img.shields.io/badge/MODEL-LightGBM-F15A24?style=for-the-badge" />
  <img src="https://img.shields.io/badge/LANGUAGE-Python-FFD93D?style=for-the-badge" />
</p>

<h2 align="center">Turning Complex Business Data into Reliable Matches</h2>

<p align="center">
  An intelligent entity resolution pipeline designed to identify matching business records across multiple data sources.
</p>

---

## 📌 Overview

Business Entity Resolution is a machine learning solution developed for the **Amazon ML Challenge 2026**.

The system identifies records referring to the same business across different datasets using text normalization, candidate generation, feature engineering, and LightGBM classification.

## ⚙️ Resolution Pipeline

1. Dataset inspection
2. Text normalization
3. Blocking and candidate generation
4. Training-pair creation
5. Feature engineering
6. LightGBM model training
7. Validation and threshold selection
8. Test candidate generation
9. Test prediction
10. Submission formatting and validation

## 🧠 Model Information

| Component | Details |
|---|---|
| Algorithm | LightGBM |
| Version | 4.7.0 |
| Prediction threshold | 0.81 |
| License | MIT |

## 📊 Validation Performance

| Metric | Result |
|---|---:|
| F0.5 Score | 0.9811 |
| Precision | 0.9894 |
| Recall | 0.9491 |
| AUC | 0.99936 |

*These are validation-set results, not the hidden test-set score.*

## 📁 Final Test Output

- `matching_results.tsv` — one row per Source 1 entity with comma-separated matched IDs.
- `candidate_pairs.tsv` — candidate IDs supplied to the matching stage.

Both files contain all **1,732,544 required Source 1 entities**.

## ✅ Validation

The official challenge validator was run with ID checking enabled.

**Result: PASS — no blocking issues found.**

Additional checks confirmed:
- No duplicate Source 1 rows.
- No duplicate IDs.
- Every predicted ID exists in the test Source 2/Source 3 data.

## 🚀 Reproducibility

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Source code is available in the `src` directory.

The solution uses the challenge-provided datasets and does not use external business-entity lookup.

---

<p align="center">
  <b>🟡 Precision in Data. 🧡 Intelligence in Matching.</b>
</p>
