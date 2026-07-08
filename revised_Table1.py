# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 20:35:49 2026

@author: vshas
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
dfo = build_df_ov(df, max_missing_items=1)


df = dfo.copy()
TOTAL = len(df)
def harmonize_sex(row):
    s, g = row["sex"], row["gender"]
    if pd.notna(s):
        return s
    if pd.notna(g):
        g = str(g)
        if g in ("M", "0", "0.0"):  return "Male"
        if g in ("F", "1", "1.0"):  return "Female"
    return np.nan
 
df["sex_h"] = df.apply(harmonize_sex, axis=1)
 
RACEUS_LABELS = {
    0: "White", 1: "Black/African American", 2: "Asian",
    3: "American Indian/Alaska Native",
    4: "Native Hawaiian/Pacific Islander", 5: "Other/Multiracial",
}
 
STUDY_ORDER = ["ACAP", "MHIP", "MIOS", "Max"]
ORDER = ["Overall"] + STUDY_ORDER
 
def pct(n, d):
    return f"{n} ({n/d*100:.1f}%)" if d else "—"
 
def mean_sd(s):
    s = s.dropna()
    return f"{s.mean():.1f} ({s.std():.1f})" if len(s) else "—"
 
def median_iqr(s, dec=0):
    s = s.dropna()
    if not len(s): return "—"
    return f"{s.median():.{dec}f} [{s.quantile(.25):.{dec}f}–{s.quantile(.75):.{dec}f}]"
 
def value_range(s, dec=1):
    s = s.dropna()
    return f"{s.min():.{dec}f}–{s.max():.{dec}f}" if len(s) else "—"
 
def compute(d):
    out = {}
    out["Sample size, n (% of cohort)"] = f"{len(d):,} ({len(d)/TOTAL*100:.1f}%)"
 
    out["Any TBI, n (%)"] = pct((d["clin_group"] == "TBI").sum(), len(d))
    out["Orthopedic injury (OI), n (%)"] = pct((d["clin_group"] == "OI").sum(), len(d))
 
    out["Age — Mean (SD)"] = mean_sd(d["age"])
    out["Age — Median [IQR]"] = median_iqr(d["age"], dec=1)
    out["Age — Range"] = value_range(d["age"])
    n_age = d["age"].notna().sum()
 
    out["Male, n (%)"] = pct((d["sex_h"] == "Male").sum(), d["sex_h"].notna().sum())
    out["Female, n (%)"] = pct((d["sex_h"] == "Female").sum(), d["sex_h"].notna().sum())
 
    rus = d["raceUS"].dropna()
    for code, label in RACEUS_LABELS.items():
        out[f"Race (US): {label}, n (%)"] = pct((rus == code).sum(), len(rus)) if len(rus) else "—"
 
    eth = d["ethUS"].dropna()
    out["Ethnicity (US): Hispanic/Latino, n (%)"] = pct((eth == 1).sum(), len(eth)) if len(eth) else "—"
    out["Ethnicity (US): Non-Hispanic/Latino, n (%)"] = pct((eth == 0).sum(), len(eth)) if len(eth) else "—"
 
    dsi = d["days_since_injury"]
    out["Days since injury — Median [IQR]"] = median_iqr(dsi)
    out["Days since injury — Mean (SD)"] = mean_sd(dsi)
    out["Days since injury — n with data"] = str(dsi.notna().sum())
 
    return out
 
rows = {}
for name in ORDER:
    sub = df if name == "Overall" else df[df["study"] == name]
    rows[name] = compute(sub)
 
result = pd.DataFrame(rows)
result.index.name = "Characteristic"

#%%

# Count NaNs
N = len(dfo)
# ── Column groups ────────────────────────────────────────────────────────────
hbi_child_items  = [c for c in dfo.columns if c.startswith('HBIc') and 'sum' not in c]
hbi_parent_items = [c for c in dfo.columns if c.startswith('HBIp') and 'sum' not in c]
mpcs_child_items = [c for c in dfo.columns if c.startswith('MPCSc') and 'sum' not in c]
mpcs_parent_items= [c for c in dfo.columns if c.startswith('MPCSp') and 'sum' not in c]
sum_cols         = ['HBIc_sum', 'HBIp_sum', 'MPCSc_sum', 'MPCSp_sum']
 
groups = {
    'HBI child items (HBIc01–20)'        : hbi_child_items,
    'HBI parent items (HBIp01–20)'       : hbi_parent_items,
    'MPCS child items (MPCSc01–15)'      : mpcs_child_items,
    'MPCS parent items (MPCSp01–15)'     : mpcs_parent_items,
    'Sum scores (regression outcomes)'   : sum_cols,
}
 
# ── 1. Per-column missingness ────────────────────────────────────────────────
col_rows = []
for grp_name, cols in groups.items():
    for c in cols:
        n_miss = int(dfo[c].isna().sum())
        col_rows.append({
            'Group'   : grp_name,
            'Column'  : c,
            'N missing': n_miss,
            'N present': N - n_miss,
            'Missing %': round(n_miss / N * 100, 1),
        })
 
df_col = pd.DataFrame(col_rows)
print("=" * 70)
print("PER-COLUMN MISSINGNESS")
print("=" * 70)
print(df_col.to_string(index=False))
 
# ── 2. Per-group summary ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("GROUP SUMMARY  (matrix = rows × items in each group)")
print("=" * 70)
print(f"{'Group':<45} {'Items':>5} {'Matrix':>8} {'NaN total':>10} {'% missing':>10}")
print("-" * 70)
for grp_name, cols in groups.items():
    sub    = dfo[cols]
    n_items= len(cols)
    matrix = N * n_items
    n_nan  = int(sub.isna().sum().sum())
    pct    = round(n_nan / matrix * 100, 1) if matrix else 0
    print(f"{grp_name:<45} {n_items:>5} {matrix:>8,} {n_nan:>10,} {pct:>9.1f}%")
 
# ── 3. Per-row missingness for each group ────────────────────────────────────
print("\n" + "=" * 70)
print("PER-ROW MISSING COUNT DISTRIBUTION (how many items missing per participant)")
print("=" * 70)
for grp_name, cols in groups.items():
    row_miss = dfo[cols].isna().sum(axis=1)
    print(f"\n{grp_name}")
    vc = row_miss.value_counts().sort_index()
    # show only counts 0 and >0 collapsed
    n_complete  = int((row_miss == 0).sum())
    n_any_miss  = int((row_miss  > 0).sum())
    n_all_miss  = int((row_miss == len(cols)).sum())
    print(f"  Rows with 0 missing (complete) : {n_complete:>5}  ({n_complete/N*100:.1f}%)")
    print(f"  Rows with ≥1 missing           : {n_any_miss:>5}  ({n_any_miss/N*100:.1f}%)")
    print(f"  Rows with ALL missing           : {n_all_miss:>5}  ({n_all_miss/N*100:.1f}%)")
    if n_any_miss > 0:
        print(f"  Distribution of missing counts among those with ≥1 missing:")
        partial = vc[vc.index > 0]
        for k, v in partial.items():
            print(f"    {int(k):>3} items missing: {v:>5} rows")
 
# ── 4. Overall matrix total ──────────────────────────────────────────────────
all_cols = sum(groups.values(), [])
all_cols_unique = list(dict.fromkeys(all_cols))   # preserve order, dedupe
total_matrix = N * len(all_cols_unique)
total_nan    = int(dfo[all_cols_unique].isna().sum().sum())
print("\n" + "=" * 70)
print("GRAND TOTAL (all unique instrument columns combined)")
print("=" * 70)
print(f"  Rows (participants) : {N:,}")
print(f"  Columns (items)     : {len(all_cols_unique):,}")
print(f"  Matrix size         : {total_matrix:,}")
print(f"  Total NaN           : {total_nan:,}")
print(f"  Overall missing %   : {total_nan/total_matrix*100:.1f}%")