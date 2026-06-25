# -*- coding: utf-8 -*-
import os 
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.metrics import roc_auc_score
import umap
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from peer_funcs import load_plotting_settings, binary_scores, save_as_pickle, load_pickle
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from itertools import product
import matplotlib.pyplot as plt

from collections import Counter
import pickle as pkl
import mord
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, mean_absolute_error,roc_auc_score, accuracy_score
from scipy.stats import spearmanr
import warnings
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed

def weight_features(source_features, source_scores):
    # weight a semantic feature set by empirical scores for those items
    
    nfeats = np.shape(source_features)[1]
    
    weighted_matrix = np.zeros((len(source_scores.index), nfeats))
    
    # for every person
    for j,ind in enumerate(source_scores.index):
        
        # create a blank weighted vector object
        w_vec = np.zeros(nfeats)
        
        # get their predictive feature scores on A
        source_row_scores = source_scores.loc[ind].values
    
        # loop over items and sum weighted scores
        for i,vec in enumerate(source_features):
            w_vec = w_vec + source_row_scores[i]*vec
            
        weighted_matrix[j,:] = w_vec
        
    return weighted_matrix

# load_plotting_settings()
# os.chdir(r'C:\Users\u6029515\Box\UU_EK\Projects\Harmonization\Pediatric\Analysis')
# os.chdir(r"C:\Users\u6029515\Box\UU_EK\Projects\Harmonization\Pediatric\Analysis\embed_dict_models")
os.chdir(r"C:\Users\vshas\Box\Harmonization\Pediatric\Analysis")
# score_dict = load_pickle("score_dict.p")
# text_dict = load_pickle("text_dict.p")
embed_dict = load_pickle("./embed_dict_models/embed_dict.p")

df = pd.read_excel("../Data/Peds TBI Harmonization_3.xlsx")

df['sexgender'] = (df['sex'].astype(str) + df['gender'].astype(str)).replace(
    {'nan0':'M','0.0nan':'M','nanM':'M','1.0nan':'F','nan1':'F','nanF':'F','nannan':'M'})
# correction for item 15 bad data for children
# df['MPCSc15'] = df['MPCSp15']

# Fix the index FIRST — this is what caused the iloc/loc inflation.
df = df.reset_index(drop=True)

# ── Item name lists ───────────────────────────────────────────────────────────
HBI_c_items  = [f'HBIc{i:02d}'  for i in range(1, 21)]   # HBI child  (20 items)
HBI_p_items  = [f'HBIp{i:02d}'  for i in range(1, 21)]   # HBI parent (20 items)
MPCS_c_items = [f'MPCSc{i:02d}' for i in range(1, 16)]   # M-PCSI child  (15 items)
MPCS_p_items = [f'MPCSp{i:02d}' for i in range(1, 16)]   # M-PCSI parent (15 items)

# ── Feature sums (each sum uses its OWN items — the swap is fixed) ─────────────
ITEM_GROUPS = {
    'HBIc_sum':  HBI_c_items,
    'HBIp_sum':  HBI_p_items,
    'MPCSc_sum': MPCS_c_items,
    'MPCSp_sum': MPCS_p_items,
}
score_sum_vars = list(ITEM_GROUPS)
for name, cols in ITEM_GROUPS.items():
    df[name] = df[cols].sum(axis=1)

# ── "Usable" masks: at most 1 missing item per required block ─────────────────
def few_missing(cols):
    return df[cols].isnull().sum(axis=1) <= 1

hbi_ok  = few_missing(HBI_p_items)  & few_missing(HBI_c_items)
mpcs_ok = few_missing(MPCS_p_items) & few_missing(MPCS_c_items)

# ── Selection by boolean mask (cannot double-count, index-safe) ───────────────
df_HBI  = df[hbi_ok].drop_duplicates('subject_ID', keep='first')
df_MPCS = df[mpcs_ok].drop_duplicates('subject_ID', keep='first')
dfo     = df[hbi_ok & mpcs_ok].drop_duplicates('subject_ID', keep='first').copy()

#%%

# Analysis
# ── Config ────────────────────────────────────────────────────────────────────
 
warnings.filterwarnings("ignore", category=DeprecationWarning, module="mord")
warnings.filterwarnings("ignore", message=".*disp.*iprint.*L-BFGS-B.*")
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
 

# ── Shared config ─────────────────────────────────────────────────────────────
MODEL_NAME         = 'all-mpnet-base-v2'
NUM_TRIES          = 10
# TRAIN_SIZES        = [0.50, 0.60, 0.70, 0.80, 0.90]#
TRAIN_SIZES        = [0.70]
N_ESTIMATORS       = 500
SMOTE_FLOOR        = 100
USE_SMOTE          = True
OUTER_JOBS         = -1
 
# primary-specific
BINARIZE_THRESHOLD = 1            # y > 1 -> 1
N_SIMILAR_FEATURES = 3            # logistic uses the n most similar source items
N_PCA_COMPONENTS   = 10           # for the combined RF features
# secondary-specific
ORDINAL_ALPHA      = 1.0
ORDINAL_LABELS     = [0, 1, 2, 3]
 
 
def _safe_ext(model_name): return model_name.split("/")[-1]
def _make_rf(seed):
    return RandomForestClassifier(n_estimators=N_ESTIMATORS, n_jobs=1, random_state=seed)
 
 
def weight_features(source_features, source_scores_values):
    """Score-weighted sum of item embeddings: (n,n_items)@(n_items,nfeats)->(n,nfeats)."""
    return source_scores_values @ source_features
 
 
def select_similar_items(source_features, target_emb, n):
    """Indices of the n source items most cosine-similar to the target item."""
    sims = (source_features @ target_emb) / (
        np.linalg.norm(source_features, axis=1) * np.linalg.norm(target_emb) + 1e-12)
    return np.argsort(sims)[::-1][:n]
 
 
def _multiclass_smote(X, y, floor=SMOTE_FLOOR, random_state=0):
    """Oversample minority classes up to `floor` (works for binary and 0-3)."""
    cnt = Counter(y); maj = max(cnt.values()); target = min(floor, maj)
    strat = {c: target for c in cnt if cnt[c] < target}
    if not strat:
        return X, y, False
    smallest = min(cnt.values())
    if smallest < 2:
        return X, y, False
    sm = SMOTE(sampling_strategy=strat, k_neighbors=min(5, smallest - 1),
               random_state=random_state)
    Xr, yr = sm.fit_resample(X, y)
    return Xr, yr, True
 
def _binaryclass_smote(X, y, threshold=98, mult=2.5, random_state=0):
    """Replica of the original binarized SMOTE: if class-1 count < threshold,
    oversample class 1 to threshold*mult (=245) cases via a float ratio.
    y must be binarized {0,1}. Returns (X_res, y_res, did_resample)."""
    counts = pd.Series(y).value_counts()          # label-indexed (class1/class0)
    ones  = counts.get(1, 0)
    zeros = counts.get(0, 0)
    if ones < threshold:
        target_ratio = threshold * mult / zeros   # 245 / #zeros
        cur_ratio = ones / zeros if zeros else np.inf
        if target_ratio <= cur_ratio or target_ratio > 1.0:   # SMOTE would raise
            target_ratio = min(max(target_ratio, cur_ratio + 1e-9), 1.0)
        over = SMOTE(sampling_strategy=target_ratio, random_state=random_state)
        Xr, yr = over.fit_resample(X, y)
        return Xr, yr, True
    return X, y, False

def _complete_case(source_scores, target_col_values):
    """Rows with no NaN in source items or target. Returns (Xs, y_raw, mask) or None."""
    mask = source_scores.notna().all(axis=1) & pd.notna(target_col_values)
    if mask.sum() == 0:
        return None
    Xs = source_scores[mask.values]
    y_raw = pd.Series(target_col_values)[mask.values].astype(int)
    return Xs, y_raw, mask
 
 
def binarize_target(y_raw):
    """Scale-aware binarization.
    MPCS items are already 0/1 -> pass through unchanged.
    HBI items are 0-3          -> present/absent split at > BINARIZE_THRESHOLD."""
    y = np.asarray(y_raw).astype(int)
    if set(np.unique(y)).issubset({0, 1}):
        return y                                  # already binary (MPCS)
    return (y > BINARIZE_THRESHOLD).astype(int)   # 0-3 (HBI) -> 0/1
 
 
def is_ordinal_target(y_raw):
    """True only for genuinely multi-level (0-3) targets, e.g. HBI.
    MPCS items are 0/1 and are NOT ordinal — secondary analysis skips them."""
    return not set(np.unique(np.asarray(y_raw).astype(int))).issubset({0, 1})
 
 
def _agg(rec, store, prefix):
    for mt, vals in store.items():
        a = np.asarray(vals, float) if len(vals) else np.array([np.nan])
        tag = f'{mt}_{prefix}'
        rec[f'mean_{tag}'] = a.mean(); rec[f'std_{tag}'] = a.std()
        rec[f'min_{tag}']  = a.min();  rec[f'max_{tag}'] = a.max()
 
# ══════════════════════════════════════════════════════════════════════════════
# PRIMARY ANALYSIS  (binarized)
# ══════════════════════════════════════════════════════════════════════════════
def _primary_item(item_idx, target_col, source_features, source_scores,
                  tcv, target_emb, num_tries, train_sizes, use_smote):
    cc = _complete_case(source_scores, tcv)
    if cc is None:
        return [], 0
    Xs, y_raw, mask = cc
    y_bin = binarize_target(y_raw.values)
    if len(np.unique(y_bin)) < 2:
        return [], 0
 
    # SMOTE the raw source scores against the BINARY target
    if use_smote:
        # Xb, yb, did = _multiclass_smote(Xs.values, y_bin)   # multiclass floor variant
        Xb, yb, did = _binaryclass_smote(Xs.values, y_bin)      # original binarized logic (98 * 2.5)
        Xb, yb = np.asarray(Xb), np.asarray(yb)
    else:
        Xb, yb, did = Xs.values, y_bin, False
 
    # logistic features: n most-similar source items (raw scores)
    sel_idx = select_similar_items(source_features, target_emb, N_SIMILAR_FEATURES)
    X_sim   = Xb[:, sel_idx]
    # RF features:
    #   X_comb = raw + item_sum + PCA(weighted)   -> rf      (weighted-feature model)
    #   X_raw  = raw scores only                  -> rf_raw  (ablation: no weighted feats)
    weighted = weight_features(source_features, Xb)
    comb     = np.concatenate([weighted * target_emb[np.newaxis, :], weighted], axis=1)
    emb_pca  = PCA(n_components=N_PCA_COMPONENTS).fit_transform(comb)
    X_comb   = np.concatenate([Xb, Xb.sum(1, keepdims=True), emb_pca], axis=1)
    X_raw    = Xb
 
    out = []
    for train_size in train_sizes:
        ts = 1.0 - train_size
        M = {'logit':  {'acc': [], 'bal_acc': [], 'qwk': [], 'roc_auc': []},
             'logit_all':  {'acc': [], 'bal_acc': [], 'qwk': [], 'roc_auc': []},
             'rf':     {'acc': [], 'bal_acc': [], 'qwk': [], 'roc_auc': []},
             'rf_raw': {'acc': [], 'bal_acc': [], 'qwk': [], 'roc_auc': []}}
        for t in range(num_tries):
            seed = (item_idx + 1) * (t + 1) * int(train_size * 100)
            strat = yb if min(Counter(yb).values()) >= 2 else None
            itr, ite = train_test_split(np.arange(len(yb)),
                                        test_size=ts,
                                        random_state=seed, 
                                        stratify=strat)
            ytr, yte = yb[itr], yb[ite]
            two_classes = len(np.unique(yte)) == 2
            
            lr = LogisticRegression(max_iter=1000).fit(X_raw[itr], ytr)
            ypl = lr.predict(X_raw[ite])
            M['logit_all']['acc'].append(accuracy_score(yte, ypl))
            M['logit_all']['bal_acc'].append(balanced_accuracy_score(yte, ypl))
            M['logit_all']['qwk'].append(cohen_kappa_score(yte, ypl, labels=[0, 1], weights='quadratic'))
            if two_classes:
                M['logit_all']['roc_auc'].append(roc_auc_score(yte, lr.predict_proba(X_raw[ite])[:, 1]))
 
            lr = LogisticRegression(max_iter=1000).fit(X_sim[itr], ytr)
            ypl = lr.predict(X_sim[ite])
            M['logit']['acc'].append(accuracy_score(yte, ypl))
            M['logit']['bal_acc'].append(balanced_accuracy_score(yte, ypl))
            M['logit']['qwk'].append(cohen_kappa_score(yte, ypl, labels=[0, 1], weights='quadratic'))
            if two_classes:
                M['logit']['roc_auc'].append(roc_auc_score(yte, lr.predict_proba(X_sim[ite])[:, 1]))
 
            # RF with weighted features
            rf = _make_rf(seed).fit(X_comb[itr], ytr)
            ypr = rf.predict(X_comb[ite])
            M['rf']['acc'].append(accuracy_score(yte, ypr))
            M['rf']['bal_acc'].append(balanced_accuracy_score(yte, ypr))
            M['rf']['qwk'].append(cohen_kappa_score(yte, ypr, labels=[0, 1], weights='quadratic'))
            if two_classes:
                M['rf']['roc_auc'].append(roc_auc_score(yte, rf.predict_proba(X_comb[ite])[:, 1]))
 
            # RF ablation: raw scores only (same target, same split, same SMOTE)
            rfr = _make_rf(seed).fit(X_raw[itr], ytr)
            yprr = rfr.predict(X_raw[ite])
            M['rf_raw']['acc'].append(accuracy_score(yte, yprr))
            M['rf_raw']['bal_acc'].append(balanced_accuracy_score(yte, yprr))
            M['rf_raw']['qwk'].append(cohen_kappa_score(yte, yprr, labels=[0, 1], weights='quadratic'))
            if two_classes:
                M['rf_raw']['roc_auc'].append(roc_auc_score(yte, rfr.predict_proba(X_raw[ite])[:, 1]))
 
        rec = {'target_item': target_col, 'item_idx': item_idx, 'train_size': train_size,
               'n_used': int(mask.sum()), 'pct_ones': float(y_bin.mean()),
               'sim_items': ",".join(map(str, Xs.columns[sel_idx]))}
        _agg(rec, M['logit'],  'logit')        
        _agg(rec, M['logit_all'],  'logit_all')
        _agg(rec, M['rf'],     'rf')
        _agg(rec, M['rf_raw'], 'rf_raw')
        out.append(rec)
    return out, int(did)
 
 
def run_primary_analysis(dfo, source_name, source_items, target_name, target_items,
                         embed_dict, num_tries=NUM_TRIES, train_sizes=TRAIN_SIZES,
                         use_smote=USE_SMOTE, outer_jobs=OUTER_JOBS):
    sf = embed_dict[source_name]
    ss = dfo[source_items].copy()
    jobs = Parallel(n_jobs=outer_jobs)(
        delayed(_primary_item)(i, c, sf, ss, dfo[c].values,
                               embed_dict[target_name][i], num_tries, train_sizes, use_smote)
        for i, c in enumerate(target_items))
    recs, imb = [], 0
    for r, d in jobs:
        recs.extend(r); imb += d
    return pd.DataFrame(recs), imb
 
# ══════════════════════════════════════════════════════════════════════════════
# SECONDARY ANALYSIS  (ordinal 0-3, raw features)
# ══════════════════════════════════════════════════════════════════════════════
def _binary_collapse(clf, Xte, y_te):
    """Collapse a 0-3 classifier's output to the 0/1 task. Returns (acc, bacc, qwk, roc)."""
    yp = clf.predict(Xte)
    ybt = (y_te > BINARIZE_THRESHOLD).astype(int)
    ybp = (yp   > BINARIZE_THRESHOLD).astype(int)
    acc  = accuracy_score(ybt, ybp)
    bacc = balanced_accuracy_score(ybt, ybp)
    qwk  = cohen_kappa_score(ybt, ybp, labels=[0, 1], weights='quadratic')
    roc  = np.nan
    if len(np.unique(ybt)) == 2 and hasattr(clf, 'predict_proba'):
        proba = clf.predict_proba(Xte)
        cols = [k for k, c in enumerate(clf.classes_) if c > BINARIZE_THRESHOLD]
        if cols:
            roc = roc_auc_score(ybt, proba[:, cols].sum(axis=1))
    return yp, acc, bacc, qwk, roc
 
 
def _secondary_item(item_idx, target_col, source_scores, tcv,
                    num_tries, train_sizes, use_smote):
    cc = _complete_case(source_scores, tcv)
    if cc is None:
        return [], 0
    Xs, y_raw, mask = cc
    if y_raw.nunique() < 2:
        return [], 0
    if not is_ordinal_target(y_raw.values):      # skip MPCS (already 0/1) — not ordinal
        return [], 0
 
    if use_smote:
        Xb, yb, did = _multiclass_smote(Xs.values, y_raw.values)
        Xb, yb = np.asarray(Xb), np.asarray(yb)
    else:
        Xb, yb, did = Xs.values, y_raw.values, False
 
    out = []
    for train_size in train_sizes:
        ts = 1.0 - train_size
        M = {m: {'acc': [], 'bal_acc': [], 'qwk': [],
                 'acc_bin': [], 'balacc_bin': [], 'qwk_bin': [], 'roc_bin': []}
             for m in ('ord', 'rf')}
        for t in range(num_tries):
            seed = (item_idx + 1) * (t + 1) * int(train_size * 100)
            strat = yb if min(Counter(yb).values()) >= 2 else None
            itr, ite = train_test_split(np.arange(len(yb)), test_size=ts,
                                        random_state=seed, stratify=strat)
            ytr, yte = yb[itr], yb[ite]
 
            for name, clf in (('ord', mord.LogisticAT(alpha=ORDINAL_ALPHA)),
                              ('rf',  _make_rf(seed))):
                clf.fit(Xb[itr], ytr)
                yp, acc_b, bacc, qwk_b, roc_b = _binary_collapse(clf, Xb[ite], yte)
                M[name]['acc'].append(accuracy_score(yte, yp))
                M[name]['bal_acc'].append(balanced_accuracy_score(yte, yp))
                M[name]['qwk'].append(cohen_kappa_score(
                    yte, yp, labels=ORDINAL_LABELS, weights='quadratic'))
                M[name]['acc_bin'].append(acc_b)
                M[name]['balacc_bin'].append(bacc)
                M[name]['qwk_bin'].append(qwk_b)
                M[name]['roc_bin'].append(roc_b)
 
        rec = {'target_item': target_col, 'item_idx': item_idx, 'train_size': train_size,
               'n_used': int(mask.sum()), 'n_levels_raw': int(y_raw.nunique())}
        _agg(rec, M['ord'], 'ord')
        _agg(rec, M['rf'],  'rf')
        out.append(rec)
    return out, int(did)
 
 
def run_secondary_analysis(dfo, source_name, source_items, target_name, target_items,
                           embed_dict, num_tries=NUM_TRIES, train_sizes=TRAIN_SIZES,
                           use_smote=USE_SMOTE, outer_jobs=OUTER_JOBS):
    ss = dfo[source_items].copy()
    jobs = Parallel(n_jobs=outer_jobs)(
        delayed(_secondary_item)(i, c, ss, dfo[c].values, num_tries, train_sizes, use_smote)
        for i, c in enumerate(target_items))
    recs, imb = [], 0
    for r, d in jobs:
        recs.extend(r); imb += d
    return pd.DataFrame(recs), imb
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Separate executions
# ══════════════════════════════════════════════════════════════════════════════
def _load_pairs():
    pcsi = [('MPCS_p', MPCS_p_items), ('MPCS_c', MPCS_c_items)]
    hbi  = [('HBI_p',  HBI_p_items),  ('HBI_c',  HBI_c_items)]
    return list(product(pcsi, hbi)) + list(product(hbi, pcsi))
 
 
def execute_primary(dfo, embed_dict, verbose=True):
    """Run primary on all pairs. Returns one tidy DataFrame (with a `pair` column)
    of mean/std/min/max-across-attempts metrics for every (pair, target_item,
    train_size). Also returns the per-pair dict."""
    pairs, frames, per_pair, smote_rows = _load_pairs(), [], {}, []
    for (sn, si), (tn, ti) in pairs:
        key = f'{sn}-{tn}'
        df_res, n_smote = run_primary_analysis(dfo, sn, si, tn, ti, embed_dict)
        smote_rows.append({'pair': key, 'source': sn, 'target': tn,
                           'n_smote': n_smote, 'n_target_items': len(ti)})
        if not df_res.empty:
            df_res = df_res.copy(); df_res.insert(0, 'pair', key)
        per_pair[key] = df_res
        frames.append(df_res)
        if verbose:
            m = df_res.mean(numeric_only=True) if not df_res.empty else {}
            print(f"[PRIMARY] {key}  "
                  "Mean of acc"
                  
                  f"acc logit={m.get('mean_acc_logit', float('nan')):.3f} "
                  f"logit_all={m.get('mean_acc_logit_all', float('nan')):.3f} "
                  f"rf={m.get('mean_acc_rf', float('nan')):.3f} "
                  f"rf_raw={m.get('mean_acc_rf_raw', float('nan')):.3f}  |  "
                  
                  "Std of acc"
                  
                  f"acc logit={m.get('std_acc_logit', float('nan')):.3f} "
                  f"logit_all={m.get('std_acc_logit_all', float('nan')):.3f} "
                  f"rf={m.get('std_acc_rf', float('nan')):.3f} "
                  f"rf_raw={m.get('std_acc_rf_raw', float('nan')):.3f}  |  "
                  
                  "Mean of bal acc"
                  
                  f"bal_acc logit={m.get('mean_bal_acc_logit', float('nan')):.3f} "
                  f"logit_all={m.get('mean_bal_acc_logit_all', float('nan')):.3f} "
                  f"rf={m.get('mean_bal_acc_rf', float('nan')):.3f} "
                  f"rf_raw={m.get('mean_bal_acc_rf_raw', float('nan')):.3f}"
            
                  "Std of bal acc"
                  f"acc logit={m.get('std_bal_acc_logit', float('nan')):.3f} "
                  f"logit_all={m.get('std_bal_acc_logit_all', float('nan')):.3f} "
                  f"rf={m.get('std_bal_acc_rf', float('nan')):.3f} "
                  f"rf_raw={m.get('std_bal_acc_rf_raw', float('nan')):.3f}  |  ")
    tidy = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    smote_summary = pd.DataFrame(smote_rows)
    return tidy, per_pair, smote_summary
 
def execute_secondary(dfo, embed_dict, verbose=True):
    """Run secondary on all pairs (skips non-ordinal MPCS targets). Returns one
    tidy DataFrame (with a `pair` column) + the per-pair dict."""
    pairs, frames, per_pair, skipped = _load_pairs(), [], {}, []
    for (sn, si), (tn, ti) in pairs:
        key = f'{sn}-{tn}'
        df_res, _ = run_secondary_analysis(dfo, sn, si, tn, ti, embed_dict)
        if df_res.empty:
            skipped.append(key)
        else:
            df_res = df_res.copy(); df_res.insert(0, 'pair', key)
        per_pair[key] = df_res
        frames.append(df_res)
        if verbose and not df_res.empty:
            m = df_res.mean(numeric_only=True)
            print(f"[SECONDARY] {key}  "
                  f"bal_acc(4-cls) ord={m.get('mean_bal_acc_ord', float('nan')):.3f} "
                  f"rf={m.get('mean_bal_acc_rf', float('nan')):.3f}  |  "
                  f"bal_acc(binary) ord={m.get('mean_balacc_bin_ord', float('nan')):.3f} "
                  f"rf={m.get('mean_balacc_bin_rf', float('nan')):.3f}")
    if verbose and skipped:
        print(f"[SECONDARY] skipped non-ordinal targets: {', '.join(skipped)}")
    non_empty = [f for f in frames if not f.empty]
    tidy = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    return tidy, per_pair

TRAIN_SIZES = [0.50, 0.60, 0.70, 0.80, 0.90]
 
LABELS = {'HBI_c': 'HBI child', 'HBI_p': 'HBI parent',
          'MPCS_c': 'M-PCSI child', 'MPCS_p': 'M-PCSI parent'}
 
# Conversion order, matching the target layout (within-reporter then cross-reporter)
WITHIN = [('HBI_c', 'MPCS_c'), ('MPCS_c', 'HBI_c'),
          ('HBI_p', 'MPCS_p'), ('MPCS_p', 'HBI_p')]
CROSS  = [('MPCS_p', 'HBI_c'), ('MPCS_c', 'HBI_p'),
          ('HBI_p', 'MPCS_c'), ('HBI_c', 'MPCS_p')]

def _cell(vals):
    """mean (SD) across items, ×100, as 'XX.X (X.X)' or '—'."""
    v = pd.Series(vals).dropna() * 100.0
    if len(v) == 0:
        return "—"
    return f"{v.mean():.1f} ({v.std():.1f})"

def summarize_primary(primary_df, train_sizes=TRAIN_SIZES, metric="bal_acc"):
    """Aggregate per-item primary results into one row per conversion.
    metric: which per-item column family to summarize ('bal_acc' or 'acc').
    Emits LR, RF (weighted), and RF_raw (ablation) cells per train size."""
    rows = []
    for group, convs in [("Within-reporter", WITHIN), ("Cross-reporter", CROSS)]:
        for src, tgt in convs:
            key = f"{src}-{tgt}"
            dfp = primary_df[primary_df["pair"] == key]
            row = {"group": group, "source": LABELS[src], "target": LABELS[tgt], "STS": ""}
            for ts in train_sizes:
                sub = dfp[np.isclose(dfp["train_size"], ts)]
                p = int(ts * 100)
                row[f"LR_{p}"]     = _cell(sub[f"mean_{metric}_logit"])
                row[f"LR_all_{p}"]     = _cell(sub[f"mean_{metric}_logit_all"])
                row[f"RF_{p}"]     = _cell(sub[f"mean_{metric}_rf"])
                row[f"RF_raw_{p}"] = _cell(sub[f"mean_{metric}_rf_raw"])
            rows.append(row)
    return pd.DataFrame(rows)
 
from scipy import stats
ORDER  = WITHIN + CROSS

def welch_lr_all(primary_df):
    """One-sided Welch test: is LR_all > pooled (LR sim + RF wtd + RF raw)?
    Run separately for accuracy and balanced accuracy."""
    def pair_mean(s, t, col):
        return primary_df[primary_df['pair'] == f'{s}-{t}'][col].mean() * 100

    for metric, label in [('acc', 'Accuracy'), ('bal_acc', 'Balanced accuracy')]:
        g1 = np.array([pair_mean(s, t, f'mean_{metric}_logit_all') for s, t in ORDER])      # 8 LR all
        g2 = np.concatenate([[pair_mean(s, t, f'mean_{metric}_{m}') for s, t in ORDER]
                             for m in ('logit', 'rf', 'rf_raw')])                            # 24 others
        t_stat, p_two = stats.ttest_ind(g1, g2, equal_var=False)        # Welch
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2              # one-sided: g1 > g2
        print(f"{label}: LR_all mean={g1.mean():.2f} (n={len(g1)}) vs "
              f"others mean={g2.mean():.2f} (n={len(g2)}) | "
              f"Welch t={t_stat:.3f}, one-sided p={p_one:.4f} "
              f"({'significant' if p_one < 0.05 else 'n.s.'} at .05)")
        
if __name__ == "__main__":
    embed_dict = load_pickle(f"./embed_dict_models/embed_dict_{_safe_ext(MODEL_NAME)}.p")
 
    # ---- run whichever you need; they are fully independent ----
    primary_df,   primary_by_pair, primary_smote   = execute_primary(dfo, embed_dict)
    # secondary_df, secondary_by_pair = execute_secondary(dfo, embed_dict)
    
    # res_summary = summarize_primary(primary_df)
#%%
"""
Fit the FINAL LR-all converter for deployment.
For each output item, train LogisticRegression on ALL source item scores using the
FULL dataset (same binarize + SMOTE as evaluation), then pickle the list of models.
Stores actual sklearn models so the app uses the standard model.predict().
Artifact is only a few KB (logistic models are just coefficients).
Needs in scope: binarize_target, _binaryclass_smote, _complete_case, dfo, item lists.
"""
 
def fit_converter(dfo, source_items, target_items, use_smote=True):
    """Train one LogisticRegression per output item on ALL source items (full data).
    Returns (models, kept_target_items, source_items)."""
    ss = dfo[source_items]
    models, kept = [], []
    for target_col in target_items:
        cc = _complete_case(ss, dfo[target_col].values)
        if cc is None:
            continue
        Xs, y_raw, _ = cc
        y_bin = binarize_target(y_raw.values)
        if len(np.unique(y_bin)) < 2:
            continue
        X, y = Xs.values, y_bin
        if use_smote:
            X, y, _ = _binaryclass_smote(X, y)
            X, y = np.asarray(X), np.asarray(y)
        models.append(LogisticRegression(max_iter=1000).fit(X, y))
        kept.append(target_col)
    return models, kept, list(source_items)
 
 
def save_converter(path, models, target_items, source_items):
    with open(path, "wb") as f:
        pkl.dump({"models": models,
                     "target_items": target_items,
                     "source_items": source_items}, f)
 
 
# if __name__ == "__main__":
# child version: HBI <-> M-PCSI
m1, t1, s1 = fit_converter(dfo, HBI_c_items, MPCS_c_items)
save_converter("HBI-PCSI_lrall.pkl", m1, t1, s1)
 
m2, t2, s2 = fit_converter(dfo, MPCS_c_items, HBI_c_items)
save_converter("PCSI-HBI_lrall.pkl", m2, t2, s2)
 
print("saved HBI-PCSI_lrall.pkl and PCSI-HBI_lrall.pkl")
#%%
# Table 1
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
