# Predictors-of-Early-Childhood-Development-in-Nigeria-Using-Random-Forest-
UNICEF MICS data, modelling implemented in R

The project is licensed under the GNU General Public License v3.0 (GPL-3.0) and is intended for reproducible research and policy-relevant analytics.

This repository contains an R-based machine learning workflow for analyzing early childhood development outcomes in Nigeria using nationally representative MICS 2021 survey data. The project applies Random Forest classification (via the ranger engine and caret) to identify key demographic, socioeconomic, and household predictors of child development status.

The pipeline includes data import from Stata files, preprocessing and factor handling, class imbalance correction through weighting, exploratory association analysis using Cramér’s V, cross-validated model training, hyperparameter tuning, and robust evaluation with accuracy, confusion matrices, ROC curves, and AUC. Feature importance and partial dependence plots are used to support interpretability.
