# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 20:29:10 2026

@author: vshas
"""

"""
Total-score OLS regression: HBI / M-PCSI, child & parent versions.
 
Regresses each inventory total score on demographic/clinical covariates and
prints a publication-style table with * marking coefficients significant at
p < 0.01.

"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import os
from revised_preprocess import build_df_ov
# ----------------------------------------------------------------------------- 
# Config
# ----------------------------------------------------------------------------- 
os.chdir(r"C:\Users\vshas\Box\Harmonization\Pediatric\Data")

XLSX_PATH  = "Peds TBI Harmonization_3.xlsx"
SHEET_NAME = "TBI Symptom Harm"
ALPHA      = 0.01          # significance threshold for the * marker
MAX_MISSING_ITEMS = 1      # a block is "usable" if it has <= this many missing items
 
# ----------------------------------------------------------------------------- 
# Load
# ----------------------------------------------------------------------------- 
df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME).reset_index(drop=True)
df_ov = build_df_ov(df, max_missing_items=MAX_MISSING_ITEMS)

print(f"Analysis sample: n = {len(df_ov)}")
 
# ----------------------------------------------------------------------------- 
# KNN imputation of parent_edu and days_since_injury
# ----------------------------------------------------------------------------- 
KNN_FEATURES = ["age", "female", "HBIp_sum", "HBIc_sum", "MPCSp_sum", "MPCSc_sum"]
 
def knn_impute(frame, target, estimator):
    feat_ok = frame[KNN_FEATURES].notna().all(axis=1)
    known   = frame[frame[target].notna() & feat_ok]
    missing = frame[frame[target].isna()  & feat_ok]
    if len(known) == 0 or len(missing) == 0:
        return
    model = estimator(n_neighbors=1, weights="distance")
    model.fit(known[KNN_FEATURES].values, known[target].values)
    frame.loc[missing.index, target] = model.predict(missing[KNN_FEATURES].values)
 
knn_impute(df_ov, "parent_edu",        KNeighborsClassifier)
knn_impute(df_ov, "days_since_injury", KNeighborsRegressor)
 
# ----------------------------------------------------------------------------- 
# Derived covariates
# ----------------------------------------------------------------------------- 
df_ov["parent_college"] = (df_ov["parent_edu"] >= 6).astype(int)      # Bachelor's or more
df_ov["TBI"]            = (df_ov["clin_group"] == "TBI").astype(int)  # ref = OI
df_ov["white"]          = ((df_ov["raceINT"] == 13) | (df_ov["raceUS"] == 0)).astype(int)
df_ov["black"]          = ((df_ov["raceINT"] == 0)  | (df_ov["raceUS"] == 1)).astype(int)
df_ov["months_since_injury"] = df_ov["days_since_injury"] / 30
 
# ----------------------------------------------------------------------------- 
# OLS per outcome
# ----------------------------------------------------------------------------- 
PREDICTORS   = ["age", "female", "white", "black", "TBI", "months_since_injury", "parent_college"]
OUTCOME_COLS = ["HBIp_sum", "HBIc_sum", "MPCSp_sum", "MPCSc_sum"]
 
X = sm.add_constant(df_ov[PREDICTORS], has_constant="add")
 
rows = []
for ycol in OUTCOME_COLS:
    data  = pd.concat([X, df_ov[ycol]], axis=1).dropna()
    model = sm.OLS(data[ycol], data[X.columns]).fit()
    for param in model.params.index:
        rows.append({
            "outcome":   ycol,
            "predictor": param,
            "coef":      model.params[param],
            "pvalue":    model.pvalues[param],
        })
 
df_results = pd.DataFrame(rows)
print("\nLong results (df_results):")
print(df_results.to_string(index=False))
 
# ----------------------------------------------------------------------------- 
# Publication-style wide table  ( * = p < ALPHA )
# ----------------------------------------------------------------------------- 
COL_ORDER  = ["HBIc_sum", "HBIp_sum", "MPCSc_sum", "MPCSp_sum"]
COL_LABELS = ["HBI child", "HBI parent", "PCSI child", "PCSI parent"]
ROW_MAP = [
    ("TBI",                 "TBI (ref: OI)"),
    ("age",                 "Age (years)"),
    ("female",              "Female (ref: male)"),
    ("white",               "Race: White"),
    ("black",               "Race: Black"),
    ("months_since_injury", "Months since injury"),
    ("parent_college",      "Parent's education (College or more)"),
]
 
def fmt_cell(pred, outcome):
    r = df_results[(df_results.predictor == pred) & (df_results.outcome == outcome)].iloc[0]
    return f"{r.coef:.2f}" + (" *" if r.pvalue < ALPHA else "")
 
table = pd.DataFrame(
    {lab: [fmt_cell(pred, out) for pred, _ in ROW_MAP]
     for lab, out in zip(COL_LABELS, COL_ORDER)},
    index=[disp for _, disp in ROW_MAP],
)
table.index.name = "Covariate"
 
print(f"\nTotal score OLS regression (n = {len(df_ov):,})   [* = p < {ALPHA}]\n")
print(table.to_string())