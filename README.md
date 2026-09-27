# Business Entity Resolution - Amazon ML Challenge 2026

## Overview
This project performs business entity resolution between Source 1 and Source 2/3 using normalization, blocking, feature engineering, and LightGBM.

## Pipeline
1. Dataset inspection
2. Text normalization
3. Blocking and candidate generation
4. Training-pair creation
5. Feature engineering
6. LightGBM training
7. Validation and threshold selection
8. Test candidate generation
9. Test prediction
10. Submission formatting and validation

## Model
Model: LightGBM 4.7.0
License: MIT License
Prediction threshold: 0.81

## Validation Results
F0.5: 0.9811
Precision: 0.9894
Recall: 0.9491
AUC: 0.99936

These are validation-set results and are not the hidden test-set score.

## Final Test Output
matching_results.tsv and candidate_pairs.tsv contain all 1,732,544 required Source 1 entities.

## Validation
The official challenge validator was run with ID checking enabled and reported: PASS - no blocking issues found. Safe to submit.

Additional checks confirmed no duplicate S1 rows, no duplicate IDs, and every predicted ID exists in the test S2/S3 data.

## Reproducibility
Install dependencies with: pip install -r requirements.txt
Source code is contained in the src directory.
The solution uses the challenge-provided datasets and does not use external business-entity lookup.

## Output
matching_results.tsv contains one row per Source 1 entity with comma-separated matched IDs.
candidate_pairs.tsv contains one row per Source 1 entity with the final candidate IDs supplied to the matching stage.
