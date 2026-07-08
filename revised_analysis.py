# -*- coding: utf-8 -*-
import os 
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.metrics import roc_auc_score
import umap
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from peer_funcs import load_plotting_settings, binary_scores, save_as_pickle, load_pickle
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score 
from itertools import product
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import Counter
import pickle as pkl
import mord
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, mean_absolute_error,roc_auc_score, accuracy_score
from scipy.stats import spearmanr
import warnings
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from revised_preprocess import build_df_ov

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
embed_dict = load_pickle("../Analysis/embed_dict_models/embed_dict.p")

#%%

HBI_c_items  = [f"HBIc{i:02d}"  for i in range(1, 21)]   # HBI child      (20 items)
HBI_p_items  = [f"HBIp{i:02d}"  for i in range(1, 21)]   # HBI parent     (20 items)
MPCS_c_items = [f"MPCSc{i:02d}" for i in range(1, 16)]   # M-PCSI child   (15 items)
MPCS_p_items = [f"MPCSp{i:02d}" for i in range(1, 16)]   # M-PCSI parent  (15 items)
 

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
N_PCA_COMPONENTS = 10        # try more than the default 10 to test "would more PCA help?"          # for the combined RF features
# secondary-specific
ORDINAL_ALPHA      = 1.0
ORDINAL_LABELS     = [0, 1, 2, 3]
 

LABELS = {'HBI_c': 'HBI child', 'HBI_p': 'HBI parent',
          'MPCS_c': 'M-PCSI child', 'MPCS_p': 'M-PCSI parent'}
WITHIN = [('HBI_c', 'MPCS_c'), ('MPCS_c', 'HBI_c'), ('HBI_p', 'MPCS_p'), ('MPCS_p', 'HBI_p')]
CROSS  = [('MPCS_p', 'HBI_c'), ('MPCS_c', 'HBI_p'), ('HBI_p', 'MPCS_c'), ('HBI_c', 'MPCS_p')]
ORDER  = WITHIN + CROSS
# ── Model arms: 7 models × {raw, raw_wt} = 14 ────────────────────────────────
MODEL_NAMES      = ['logit', 'rf', 'svm_lin', 'svm_rbf', 'gb', 'xgb',
                    'ensemble', 'ensemble_sts_hard', 'ensemble_sts_soft']
FEATURE_SUFFIXES = ['raw', 'raw_wt']
ALL_ARMS = [f'{m}_{f}' for m in MODEL_NAMES for f in FEATURE_SUFFIXES]
 
ARM_DESCRIPTIONS = {
    'logit_raw':       'Logistic regression on all source items (raw scores)',
    'logit_raw_wt':    'Logistic regression on raw + item_sum + PCA(weighted embeddings)',
    'rf_raw':          'Random forest on raw source items only',
    'rf_raw_wt':       'Random forest on raw + item_sum + PCA(weighted embeddings)',
    'svm_lin_raw':     'Linear-kernel SVM on all source items (standardized)',
    'svm_lin_raw_wt':  'Linear-kernel SVM on raw + PCA(weighted embeddings) (standardized)',
    'svm_rbf_raw':     'RBF-kernel SVM on all source items (standardized)',
    'svm_rbf_raw_wt':  'RBF-kernel SVM on raw + PCA(weighted embeddings) (standardized)',
    'gb_raw':          'Gradient boosting on all source items',
    'gb_raw_wt':       'Gradient boosting on raw + PCA(weighted embeddings)',
    'xgb_raw':         'XGBoost on all source items',
    'xgb_raw_wt':      'XGBoost on raw + PCA(weighted embeddings)',
    'ensemble_raw':    'Soft-voting ensemble (logit + RBF-SVM + GB), NO STS — raw items',
    'ensemble_raw_wt': 'Soft-voting ensemble (logit + RBF-SVM + GB), NO STS — raw + PCA(weighted embeddings)',
    'ensemble_sts_hard_raw':    'Hard-voting ensemble (logit + RBF-SVM + GB + STS) — raw items',
    'ensemble_sts_hard_raw_wt': 'Hard-voting ensemble (logit + RBF-SVM + GB + STS) — raw + PCA(weighted embeddings)',
    'ensemble_sts_soft_raw':    'Soft-voting ensemble (logit + RBF-SVM + GB + STS) — raw items',
    'ensemble_sts_soft_raw_wt': 'Soft-voting ensemble (logit + RBF-SVM + GB + STS) — raw + PCA(weighted embeddings)',
}

def _safe_ext(model_name): return model_name.split("/")[-1]
def _make_rf(seed):
    return RandomForestClassifier(n_estimators=N_ESTIMATORS, n_jobs=1, random_state=seed)

def _svc(kernel, seed):
    """Calibrated SVC with predict_proba (replaces deprecated SVC(probability=True))."""
    return make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(SVC(kernel=kernel, random_state=seed), ensemble=False, cv=3),
    )
 
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

def make_model(name, seed, sts_col=None, sts_max=3.0):
    """Fresh estimator for a model name (fresh per fit so nothing leaks across splits)."""
    if name == 'logit':    return LogisticRegression(max_iter=2000)
    if name == 'rf':       return _make_rf(seed)
    if name == 'svm_lin':  return _svc('linear', seed)
    if name == 'svm_rbf':  return _svc('rbf', seed)
    if name == 'gb':       return GradientBoostingClassifier(random_state=seed)
    if name == 'xgb':      return XGBClassifier(n_estimators=300, max_depth=3,
                                                learning_rate=0.1, subsample=0.9,
                                                eval_metric='logloss', random_state=seed,
                                                verbosity=0)
    if name == 'ensemble': return VotingClassifier(estimators=[
                                ('lr',  LogisticRegression(max_iter=1000)),
                                ('svm', _svc('rbf', seed)),
                                ('gb',  GradientBoostingClassifier(random_state=seed)),
                            ], voting='soft')
    if name == 'ensemble_sts_hard':          # ensemble + STS, majority (predict) vote
        est = [('lr',  LogisticRegression(max_iter=1000)),
               ('svm', _svc('rbf', seed)),
               ('gb',  GradientBoostingClassifier(random_state=seed))]
        if sts_col is not None:
            est.append(('sts', STSClassifier(col_idx=int(sts_col), max_score=sts_max)))
        return VotingClassifier(estimators=est, voting='hard')

    if name == 'ensemble_sts_soft':          # ensemble + STS, averaged-proba vote
        est = [('lr',  LogisticRegression(max_iter=1000)),
               ('svm', _svc('rbf', seed)),
               ('gb',  GradientBoostingClassifier(random_state=seed))]
        if sts_col is not None:
            est.append(('sts', STSClassifier(col_idx=int(sts_col), max_score=sts_max)))
        return VotingClassifier(estimators=est, voting='soft')
    raise ValueError(name)
 
 
def _score(store, yte, ypred, yproba, two_classes):
    store['acc'].append(accuracy_score(yte, ypred))
    store['bal_acc'].append(balanced_accuracy_score(yte, ypred))
    store['qwk'].append(cohen_kappa_score(yte, ypred, labels=[0, 1], weights='quadratic'))
    store['f1'].append(f1_score(yte, ypred, zero_division=0))          # <-- new
    if two_classes and yproba is not None:
        store['roc_auc'].append(roc_auc_score(yte, yproba))
 
class STSClassifier(ClassifierMixin, BaseEstimator):
    """Zero-shot STS as an ensemble member (same rule as execute_sts_baseline):
    read the single most-similar source item's raw score (column `col_idx` of X)
    and vote with it. predict() thresholds raw/max at 0.5 -> reproduces
    binarize_target (M-PCSI max_score=1, HBI max_score=3). predict_proba() exposes
    the graded raw score in [0,1] so it can take part in *soft* voting."""
    def __init__(self, col_idx=0, max_score=3.0):
        self.col_idx = col_idx
        self.max_score = max_score
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self
    def _p1(self, X):
        col = np.asarray(X)[:, self.col_idx].astype(float)
        return np.clip(col / float(self.max_score), 0.0, 1.0)
    def predict(self, X):
        return (self._p1(X) >= 0.5).astype(int)
    def predict_proba(self, X):
        """
        STS's estimate of P(the target's binary label is 1) — 
        and STS computes it without ever looking at the target. 
        At prediction time STS only sees the source score. 
        It's trying to predict the target, so it can't use the target to judge 
        a "match."
        Example: STS sees "this person scored 2 on the matched source item" 
        and turns that one number into a guess about the target's positive/negative label.
        Now the part that resolves your intuition: 0.667 is a confidence, 
        not the final class. The prediction is that confidence thresholded at 0.5. 
        Since 0.667 ≥ 0.5, STS's actual predicted class is 1. And the target's binary 
        label for a raw score of 2 is binarize_target(2) = (2 > 1) = 1. 
        So STS predicts 1, the truth is 1 — it is correct.
        
        The proba is just a graded confidence within that — and it only hits 1.0 
        at the top of the scale (3), because 1.0 means "as positive as this scale goes."
        A 2 is positive but not maxed out, so it lands at 0.667: firmly on the "1" side,
        just not with maximum confidence.
        
        Why keep that gradation instead of snapping 2 and 3 both to 1.0? 
        That's the deliberate choice. You could make it a hard proba — 
        anything ≥ 2 → 1.0, anything ≤ 1 → 0.0 — 
        and the hard predictions would be identical. 
        But then a person who scored 3 and a person who scored 2 would cast the 
        exact same vote, and a 0 and a 1 would too. You'd be back to the coarse 0/1
        signal we saw flatten STS's ROC-AUC earlier. Dividing by the max keeps 
        "3 is more strongly positive than 2" alive, so in the soft vote a maxed-out
        source score pushes harder than a middling one.
        """
        p1 = self._p1(X)
        return np.column_stack([1.0 - p1, p1])
    
def _primary_item(item_idx, target_col, source_features, source_scores,
                  tcv, target_emb, num_tries, train_sizes, use_smote):
    cc = _complete_case(source_scores, tcv)
    if cc is None:
        return [], 0
    Xs, y_raw, mask = cc
    y_bin = binarize_target(y_raw.values)
    if len(np.unique(y_bin)) < 2:
        return [], 0
 
    if use_smote:
        Xb, yb, did = _binaryclass_smote(Xs.values, y_bin)
        Xb, yb = np.asarray(Xb), np.asarray(yb)
    else:
        Xb, yb, did = Xs.values, y_bin, False
 
    # ── two feature sets ─────────────────────────────────────────────────────
    X_raw    = Xb                                              # raw scores only
    weighted = weight_features(source_features, Xb)
    comb     = np.concatenate([weighted * target_emb[np.newaxis, :], weighted], axis=1)
    emb_pca  = PCA(n_components=N_PCA_COMPONENTS).fit_transform(comb)
    X_comb   = np.concatenate([Xb, Xb.sum(1, keepdims=True), emb_pca], axis=1)
    FEATS    = {'raw': X_raw, 'raw_wt': X_comb}
    
    # STS member for ensemble_sts_*: most-similar source item + its 0/1-vs-0-3 scale
    top1_idx = int(select_similar_items(source_features, target_emb, 1)[0])
    sts_max  = 3.0 if float(np.nanmax(Xs.iloc[:, top1_idx].values)) > 1 else 1.0
 
    out = []
    for train_size in train_sizes:
        ts = 1.0 - train_size
        M = {a: {'acc': [], 'bal_acc': [], 'qwk': [], 'roc_auc': [], 'f1': []} for a in ALL_ARMS}
        for t in range(num_tries):
            seed = (item_idx + 1) * (t + 1) * int(train_size * 100)
            strat = yb if min(Counter(yb).values()) >= 2 else None
            itr, ite = train_test_split(np.arange(len(yb)), test_size=ts,
                                        random_state=seed, stratify=strat)
            ytr, yte = yb[itr], yb[ite]
            tc = len(np.unique(yte)) == 2
 
            def fit_score(arm, est, Xtr, Xte):
                est.fit(Xtr, ytr)
                yp = est.predict(Xte)
                pr = est.predict_proba(Xte)[:, 1] if hasattr(est, 'predict_proba') else None
                _score(M[arm], yte, yp, pr, tc)
 
            for mname in MODEL_NAMES:
                for fsuf, X in FEATS.items():
                    fit_score(f'{mname}_{fsuf}',
                              make_model(mname, seed, sts_col=top1_idx, sts_max=sts_max),
                              X[itr], X[ite])
 
        rec = {'target_item': target_col, 'item_idx': item_idx, 'train_size': train_size,
               'n_used': int(mask.sum()), 'pct_ones': float(y_bin.mean())}
        for arm in ALL_ARMS:
            _agg(rec, M[arm], arm)
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
# def _binary_collapse(clf, Xte, y_te):
#     """Collapse a 0-3 classifier's output to the 0/1 task. Returns (acc, bacc, qwk, roc)."""
#     yp = clf.predict(Xte)
#     ybt = (y_te > BINARIZE_THRESHOLD).astype(int)
#     ybp = (yp   > BINARIZE_THRESHOLD).astype(int)
#     acc  = accuracy_score(ybt, ybp)
#     bacc = balanced_accuracy_score(ybt, ybp)
#     qwk  = cohen_kappa_score(ybt, ybp, labels=[0, 1], weights='quadratic')
#     roc  = np.nan
#     if len(np.unique(ybt)) == 2 and hasattr(clf, 'predict_proba'):
#         proba = clf.predict_proba(Xte)
#         cols = [k for k, c in enumerate(clf.classes_) if c > BINARIZE_THRESHOLD]
#         if cols:
#             roc = roc_auc_score(ybt, proba[:, cols].sum(axis=1))
#     return yp, acc, bacc, qwk, roc
 
 
# def _secondary_item(item_idx, target_col, source_scores, tcv,
#                     num_tries, train_sizes, use_smote):
#     cc = _complete_case(source_scores, tcv)
#     if cc is None:
#         return [], 0
#     Xs, y_raw, mask = cc
#     if y_raw.nunique() < 2:
#         return [], 0
#     if not is_ordinal_target(y_raw.values):      # skip MPCS (already 0/1) — not ordinal
#         return [], 0
 
#     if use_smote:Now
#         Xb, yb, did = _multiclass_smote(Xs.values, y_raw.values)
#         Xb, yb = np.asarray(Xb), np.asarray(yb)
#     else:
#         Xb, yb, did = Xs.values, y_raw.values, False
 
#     out = []
#     for train_size in train_sizes:
#         ts = 1.0 - train_size
#         M = {m: {'acc': [], 'bal_acc': [], 'qwk': [],
#                  'acc_bin': [], 'balacc_bin': [], 'qwk_bin': [], 'roc_bin': []}
#              for m in ('ord', 'rf')}
#         for t in range(num_tries):
#             seed = (item_idx + 1) * (t + 1) * int(train_size * 100)
#             strat = yb if min(Counter(yb).values()) >= 2 else None
#             itr, ite = train_test_split(np.arange(len(yb)), test_size=ts,
#                                         random_state=seed, stratify=strat)
#             ytr, yte = yb[itr], yb[ite]
 
#             for name, clf in (('ord', mord.LogisticAT(alpha=ORDINAL_ALPHA)),
#                               ('rf',  _make_rf(seed))):
#                 clf.fit(Xb[itr], ytr)
#                 yp, acc_b, bacc, qwk_b, roc_b = _binary_collapse(clf, Xb[ite], yte)
#                 M[name]['acc'].append(accuracy_score(yte, yp))
#                 M[name]['bal_acc'].append(balanced_accuracy_score(yte, yp))
#                 M[name]['qwk'].append(cohen_kappa_score(
#                     yte, yp, labels=ORDINAL_LABELS, weights='quadratic'))
#                 M[name]['acc_bin'].append(acc_b)
#                 M[name]['balacc_bin'].append(bacc)
#                 M[name]['qwk_bin'].append(qwk_b)
#                 M[name]['roc_bin'].append(roc_b)
 
#         rec = {'target_item': target_col, 'item_idx': item_idx, 'train_size': train_size,
#                'n_used': int(mask.sum()), 'n_levels_raw': int(y_raw.nunique())}
#         _agg(rec, M['ord'], 'ord')
#         _agg(rec, M['rf'],  'rf')
#         out.append(rec)
#     return out, int(did)
 
 
# def run_secondary_analysis(dfo, source_name, source_items, target_name, target_items,
#                            embed_dict, num_tries=NUM_TRIES, train_sizes=TRAIN_SIZES,
#                            use_smote=USE_SMOTE, outer_jobs=OUTER_JOBS):
#     ss = dfo[source_items].copy()
#     jobs = Parallel(n_jobs=outer_jobs)(
#         delayed(_secondary_item)(i, c, ss, dfo[c].values, num_tries, train_sizes, use_smote)
#         for i, c in enumerate(target_items))
#     recs, imb = [], 0
#     for r, d in jobs:
#         recs.extend(r); imb += d
#     return pd.DataFrame(recs), imb
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Separate executions
# ══════════════════════════════════════════════════════════════════════════════
def _load_pairs():
    pcsi = [('MPCS_p', MPCS_p_items), ('MPCS_c', MPCS_c_items)]
    hbi  = [('HBI_p',  HBI_p_items),  ('HBI_c',  HBI_c_items)]
    return list(product(pcsi, hbi)) + list(product(hbi, pcsi))
 
 
def execute_primary(dfo, embed_dict, verbose=True):
    """Run primary on all pairs. Returns (tidy_df, per_pair_dict, smote_summary)."""
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
            print(f"[PRIMARY] {key}   (SMOTE {n_smote}/{len(ti)} items)")
            for arm in ALL_ARMS:
                print(f"    {arm:16s} "
                      f"acc={m.get(f'mean_acc_{arm}', float('nan')):.3f}  "
                      f"bal_acc={m.get(f'mean_bal_acc_{arm}', float('nan')):.3f}")
    tidy = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    smote_summary = pd.DataFrame(smote_rows)
    return tidy, per_pair, smote_summary
 
def execute_sts_baseline(dfo, embed_dict):
    """
    Runs the zero-shot Semantic Textual Similarity (STS) baseline.
    Maps each target item to the single most similar source item and
    directly compares their raw scores. No training loop required.
    """
    pairs = _load_pairs()
    rows = []

    for (sn, si), (tn, ti) in pairs:
        sf = embed_dict[sn]
        ss = dfo[si]

        acc_bin, bacc_bin, roc_bin, f1_bin, acc_ord, bacc_ord = [], [], [], [], [], []

        for i, target_col in enumerate(ti):
            target_emb = embed_dict[tn][i]

            # 1. Get the single most similar source item (n=1)
            top1_idx = select_similar_items(sf, target_emb, n=1)[0]

            # 2. Extract the raw scores for both the best source item and target
            top1_scores = ss.iloc[:, top1_idx].values
            tcv = dfo[target_col].values

            # 3. Filter to complete cases (matching your primary logic)
            mask = pd.notna(top1_scores) & pd.notna(tcv)
            if mask.sum() == 0:
                continue

            y_true_raw = tcv[mask].astype(int)
            y_pred_raw = top1_scores[mask].astype(int)

            # --- Ordinal Metrics ---
            acc_ord.append(accuracy_score(y_true_raw, y_pred_raw))
            bacc_ord.append(balanced_accuracy_score(y_true_raw, y_pred_raw))

            # --- Binary Metrics ---
            y_true_bin = binarize_target(y_true_raw)
            y_pred_bin = binarize_target(y_pred_raw)
            
            acc_bin.append(accuracy_score(y_true_bin, y_pred_bin))
            bacc_bin.append(balanced_accuracy_score(y_true_bin, y_pred_bin))
            f1_bin.append(f1_score(y_true_bin, y_pred_bin, zero_division=0))    
            
            if len(np.unique(y_true_bin)) == 2:                                        
                roc_bin.append(roc_auc_score(y_true_bin, y_pred_raw))   
            

        # Format function to match your 'XX.X (X.X)' cell layout
        def _fmt(metrics):
            if not metrics: return "—"
            return f"{np.mean(metrics) * 100:.1f} ({np.std(metrics) * 100:.1f})"

        # Store the 4 columns for this specific pair
        rows.append({
            'pair': f"{sn}-{tn}",
            'acc_STS_bin': _fmt(acc_bin),
            'balacc_STS_bin': _fmt(bacc_bin),
            'acc_STS_ord': _fmt(acc_ord),
            'balacc_STS_ord': _fmt(bacc_ord),
            'roc_STS_bin': _fmt(roc_bin),
            'f1_STS_bin':  _fmt(f1_bin),
        })

    return pd.DataFrame(rows)
# def execute_secondary(dfo, embed_dict, verbose=True):
#     """Run secondary on all pairs (skips non-ordinal MPCS targets). Returns one
#     tidy DataFrame (with a `pair` column) + the per-pair dict."""
#     pairs, frames, per_pair, skipped = _load_pairs(), [], {}, []
#     for (sn, si), (tn, ti) in pairs:
#         key = f'{sn}-{tn}'
#         df_res, _ = run_secondary_analysis(dfo, sn, si, tn, ti, embed_dict)
#         if df_res.empty:
#             skipped.append(key)
#         else:
#             df_res = df_res.copy(); df_res.insert(0, 'pair', key)
#         per_pair[key] = df_res
#         frames.append(df_res)
#         if verbose and not df_res.empty:
#             m = df_res.mean(numeric_only=True)
#             print(f"[SECONDARY] {key}  "
#                   f"bal_acc(4-cls) ord={m.get('mean_bal_acc_ord', float('nan')):.3f} "
#                   f"rf={m.get('mean_bal_acc_rf', float('nan')):.3f}  |  "
#                   f"bal_acc(binary) ord={m.get('mean_balacc_bin_ord', float('nan')):.3f} "
#                   f"rf={m.get('mean_balacc_bin_rf', float('nan')):.3f}")
#     if verbose and skipped:
#         print(f"[SECONDARY] skipped non-ordinal targets: {', '.join(skipped)}")
#     non_empty = [f for f in frames if not f.empty]
#     tidy = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
#     return tidy, per_pair


        
# ── conversion layout (8 rows) ───────────────────────────────────────────────

 
def _cell(vals):
    """mean (SD) across items, ×100, as 'XX.X (X.X)' or '—'."""
    v = pd.Series(vals).dropna() * 100.0
    if len(v) == 0:
        return "—"
    return f"{v.mean():.1f} ({v.std():.1f})"

# def summarize_primary(primary_df, train_sizes=TRAIN_SIZES, metric="bal_acc"):
#     """Aggregate per-item primary results into one row per conversion.
#     metric: which per-item column family to summarize ('bal_acc' or 'acc').
#     Emits LR, RF (weighted), and RF_raw (ablation) cells per train size."""
#     rows = []
#     for group, convs in [("Within-reporter", WITHIN), ("Cross-reporter", CROSS)]:
#         for src, tgt in convs:
#             key = f"{src}-{tgt}"
#             dfp = primary_df[primary_df["pair"] == key]
#             row = {"group": group, "source": LABELS[src], "target": LABELS[tgt], "STS": ""}
#             for ts in train_sizes:
#                 sub = dfp[np.isclose(dfp["train_size"], ts)]
#                 p = int(ts * 100)
#                 row[f"LR_{p}"]     = _cell(sub[f"mean_{metric}_logit"])
#                 row[f"LR_all_{p}"]     = _cell(sub[f"mean_{metric}_logit_all"])
#                 row[f"RF_{p}"]     = _cell(sub[f"mean_{metric}_rf"])
#                 row[f"RF_raw_{p}"] = _cell(sub[f"mean_{metric}_rf_raw"])
#             rows.append(row)
#     return pd.DataFrame(rows)
 

def welch_arm_vs_rest(primary_df, target_arm='logit_raw'):
    """One-sided Welch test: is `target_arm` > the pool of all OTHER arms?
    Uses the 8 per-conversion means for the target vs all others pooled.
    Run separately for accuracy and balanced accuracy."""
    others = [a for a in ALL_ARMS if a != target_arm]
 
    def pair_mean(s, t, col):
        return primary_df[primary_df['pair'] == f'{s}-{t}'][col].mean() * 100
 
    for metric, label in [('acc', 'Accuracy'), ('bal_acc', 'Balanced accuracy')]:
        g1 = np.array([pair_mean(s, t, f'mean_{metric}_{target_arm}') for s, t in ORDER])
        g2 = np.concatenate([[pair_mean(s, t, f'mean_{metric}_{a}') for s, t in ORDER]
                             for a in others])
        t_stat, p_two = stats.ttest_ind(g1, g2, equal_var=False, nan_policy='omit')
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        print(f"{label}: {target_arm} mean={np.nanmean(g1):.2f} (n={len(g1)}) vs "
              f"others mean={np.nanmean(g2):.2f} (n={len(g2)}) | "
              f"Welch t={t_stat:.3f}, one-sided p={p_one:.4f} "
              f"({'significant' if p_one < 0.05 else 'n.s.'} at .05)")


def _one_table(sub_df, arms, sts_df=None):
    """Build the 8-row combined table from rows of ONE train size."""
    rows = []
    for group, convs in [('Within-reporter', WITHIN), ('Cross-reporter', CROSS)]:
        for src, tgt in convs:
            pair_key = f'{src}-{tgt}'
            sub = sub_df[sub_df['pair'] == pair_key]
            row = {'group': group, 'source': LABELS[src], 'target': LABELS[tgt]}

            # Append dynamic ML Arms
            for arm in arms:
                row[f'acc_{arm}'] = _cell(sub[f'mean_acc_{arm}'])
            for arm in arms:
                row[f'balacc_{arm}'] = _cell(sub[f'mean_bal_acc_{arm}'])
            for arm in arms:
                row[f'roc_{arm}'] = _cell(sub[f'mean_roc_auc_{arm}'])
            for arm in arms:
                row[f'f1_{arm}']  = _cell(sub[f'mean_f1_{arm}'])
            # Append static STS Baseline
            if sts_df is not None:
                sts_sub = sts_df[sts_df['pair'] == pair_key]
                if not sts_sub.empty:
                    row['acc_STS_bin'] = sts_sub['acc_STS_bin'].values[0]
                    row['balacc_STS_bin'] = sts_sub['balacc_STS_bin'].values[0]
                    row['acc_STS_ord'] = sts_sub['acc_STS_ord'].values[0]
                    row['balacc_STS_ord'] = sts_sub['balacc_STS_ord'].values[0]
                    row['roc_STS_bin'] = sts_sub['roc_STS_bin'].values[0]
                    row['f1_STS_bin']  = sts_sub['f1_STS_bin'].values[0]

            rows.append(row)
    return pd.DataFrame(rows)
 
def combined_methods_tables(primary_df,sts_df=None, out_prefix=None, verbose=True):
    """One combined (acc + balacc, all detected arms) table PER train size.
    Returns ({train_size: DataFrame}, arms). Writes one CSV per train size if out_prefix set."""
    found = [c[len('mean_bal_acc_'):] for c in primary_df.columns if c.startswith('mean_bal_acc_')]
    arms = [a for a in ALL_ARMS if a in found] + sorted(a for a in found if a not in ALL_ARMS)
    train_sizes = sorted(primary_df['train_size'].unique())
 
    if verbose:
        print(f"Arms detected ({len(arms)}): {', '.join(arms)}")
        print(f"Train sizes ({len(train_sizes)}): {train_sizes}")
        print(f"Each table: group, source, target, acc_<arm> x{len(arms)}, "
              f"balacc_<arm> x{len(arms)}  (= {len(arms)*2} metric cols)\n")
        print("Legend (cells = mean (SD) across items, x100):")
        for arm in arms:
            print(f"  acc_{arm} / balacc_{arm}: {ARM_DESCRIPTIONS.get(arm, '(no description)')}")
        print()
 
    tables = {}
    for ts in train_sizes:
        sub = primary_df[np.isclose(primary_df['train_size'], ts)]
        tbl = _one_table(sub, arms,sts_df)
        tables[ts] = tbl
        if out_prefix:
            path = f"{out_prefix}_train{int(round(ts*100))}.csv"
            tbl.to_csv(path, index=False)
            if verbose:
                print(f"train_size={ts:.2f} -> saved {path}")
    return tables, arms


if __name__ == "__main__":
    embed_dict = load_pickle(f"./embed_dict_models/embed_dict_{_safe_ext(MODEL_NAME)}.p")
    
    print("Running STS Zero-Shot Baseline...")
    sts_df = execute_sts_baseline(dfo, embed_dict)
 
    # 2. Run Primary ML analysis
    print("Running Primary Models...")
    # ---- run whichever you need; they are fully independent ----
    primary_df,   primary_by_pair, primary_smote   = execute_primary(dfo, embed_dict)
    tables, arms = combined_methods_tables(primary_df, sts_df=sts_df)
    for ts, tbl in tables.items():
        print(f"\n===== train_size = {ts:.2f} =====")
        print(tbl.to_string(index=False))
    # secondary_df, secondary_by_pair = execute_secondary(dfo, embed_dict)
  
tbl.to_csv("results_acc_70_30_pca_10.csv")
    # res_summary = summarize_primary(primary_df)
#%%

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
 
# ---------- fonts: Poppins (sans) + Lora (serif); math font carries the arrows ----------
_FD = './fonts/'
for f in ["Poppins-Regular.ttf","Poppins-Medium.ttf","Poppins-Bold.ttf","Poppins-Light.ttf","Lora-VariableFont_wght.ttf"]:
    try: fm.fontManager.addfont(_FD+f)
    except Exception: pass
SANS, SERIF = "Poppins", "Lora"
plt.rcParams["font.family"] = SANS
plt.rcParams["mathtext.fontset"] = "dejavusans"   # renders  ->  and  <->  glyphs
 
# ---------- palette (muted, warm, harmonious) ----------
INK, GREY, FAINT = "#1F1B18", "#6B6660", "#E7E3DD"
AXIS   = "#2A2622"   # dark x/y axis lines
TICKC  = "#3A3530"   # tick-label colour (shared by x & y)
TICKW  = "regular"   # tick-label weight (shared by x & y)
 
# series: key, label, marker, colour, size, emph
#   emph=True -> big star + dark outline (reserved for the highlighted best performer)
# Dropped for clarity (kept commented): SVM-lin (< SVM-rbf), XGB (< GB), Ens+STS hard (< soft)
SERIES = [
    ("logit_raw",              "Logit",        "o", "#CB6A4A",  70, False),
    ("rf_raw",                 "RF",           "s", "#E0A340",  64, False),
    # ("svm_lin_raw",          "SVM (lin)",    "^", "#3E897F",  74, False),   # dropped: lin < rbf
    ("svm_rbf_raw",            "SVM (rbf)",    "^", "#3E897F",  74, False),
    ("gb_raw",                 "GB",           "D", "#5C7EA8",  58, False),
    # ("xgb_raw",              "XGB",          "D", "#A85C7C",  58, False),   # dropped: XGB < GB
    ("ensemble_raw",           "Ensemble",     "P", "#6F6357",  78, False),
    ("ensemble_sts_soft_raw",  "Ensemble STS", "*", "#B5651D", 240, True),   # highlighted best
    # ("ensemble_sts_hard_raw","Ens+STS hard", "X", "#7E5A9B",  74, False),   # dropped: hard < soft
    ("STS_bin",                "1-STS",        "X", "#8A857D",  74, False),   # de-emphasised baseline
]
 
METRICS = {
    "acc":    dict(title="Accuracy",          ylabel="Accuracy (%)",       divisor=1.0),
    "balacc": dict(title="Balanced accuracy", ylabel="Balanced acc. (%)",  divisor=1.0),
    "roc":    dict(title="ROC\u2013AUC",      ylabel="ROC\u2013AUC",       divisor=100.0),
    "f1":     dict(title="F1 score",          ylabel="F1 score",           divisor=100.0),
}
METRIC_ORDER = ["acc", "balacc", "roc", "f1"]
 
def _num(cell):
    try: return float(str(cell).split()[0])
    except Exception: return np.nan
 
def _labels(df):
    return [rf"{s} $\rightarrow$ {t}" for s, t in zip(df["source"], df["target"])]
 
def _emph_kw(emph):
    return dict(edgecolors=("#221F1C" if emph else "white"),
                linewidths=(1.1 if emph else 0.5),
                zorder=(6 if emph else 3))
 
# ══════════════════════════════════════════════════════════════════════════════
# Stacked A4-portrait figure: 4 metrics on a shared x-axis, one legend outside
# ══════════════════════════════════════════════════════════════════════════════
def plot_stacked(df, save=None, show=False, group_caps=True):
    n = len(df); x = np.arange(n)
    off = np.linspace(-0.12, 0.12, len(SERIES))          # tight cluster within each group
 
    fig, axes = plt.subplots(len(METRIC_ORDER), 1, sharex=True,
                             figsize=(8.9, 10.2))         # slightly wider, less tall
    fig.patch.set_facecolor("white")
 
    try:
        cut = np.where(df["group"].str.lower().str.startswith("cross"))[0][0] - 0.5
    except Exception:
        cut = None
 
    for ax, metric in zip(axes, METRIC_ORDER):
        cfg = METRICS[metric]; div = cfg["divisor"]
        ax.set_facecolor("white")
        for xi in x:
            ax.axvline(xi, color=FAINT, lw=0.9, ls=":", zorder=0)
        if cut is not None:
            ax.axvline(cut, color="#D8D2C8", lw=1.1, ls=(0,(4,3)), zorder=0.5)
 
        for (key,label,mk,col,sz,emph), dx in zip(SERIES, off):
            c = f"{metric}_{key}"
            if c not in df.columns: continue
            y = df[c].map(_num).values / div
            ax.scatter(x+dx, y, marker=mk, s=sz, c=col, alpha=0.96,
                       clip_on=False, **_emph_kw(emph))
 
        ax.set_ylabel(cfg["ylabel"], fontsize=13, color=INK, labelpad=8, fontweight="bold")
        ax.tick_params(axis="both", labelsize=10.5, colors=TICKC, length=0)
        for t in ax.get_yticklabels(): t.set_fontweight(TICKW)
        ax.grid(axis="y", ls=":", lw=0.9, color=FAINT, zorder=0)
        ax.margins(y=0.16)
        for s in ("top","right"): ax.spines[s].set_visible(False)
        for s in ("left","bottom"): ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1.4)
 
    axb = axes[-1]
    axb.set_xticks(x)
    axb.set_xticklabels(_labels(df), fontsize=11, color=TICKC,
                        rotation=45, rotation_mode="anchor", ha="right")
    for t in axb.get_xticklabels(): t.set_fontweight(TICKW)
    axb.set_xlim(-0.55, n-0.45)
    axb.set_xlabel("Item conversion", fontsize=14, color=INK, labelpad=10, fontweight="bold")
 
    # header: title + subtitle on the left, legend beside them on the right
    fig.text(0.09, 0.972, "Model performance across item conversions",
             ha="left", va="top", fontsize=16.5, color=INK, fontweight="bold")
    fig.text(0.09, 0.944, "HBI $\\leftrightarrow$ M-PCSI item conversions  ·  70/30 split",
             ha="left", va="top", fontsize=11.5, color=GREY, family=SERIF, fontstyle="italic")
 
    # bordered 4x2 legend at the figure's top-right, BESIDE the title (not on a plot)
    handles = [Line2D([0],[0], marker=mk, color="none", markerfacecolor=col,
                      markeredgecolor=("#221F1C" if emph else "white"),
                      markeredgewidth=(1.1 if emph else 0.5),
                      markersize=(13 if emph else 9), label=label)
               for (key,label,mk,col,sz,emph) in SERIES]
    leg = fig.legend(handles=handles, loc="upper right", ncol=2,
                     bbox_to_anchor=(0.995, 0.995), frameon=True, framealpha=1.0,
                     borderpad=0.55, handletextpad=0.35, columnspacing=0.9,
                     labelspacing=0.4, fontsize=9, edgecolor=FAINT)
    leg.get_frame().set_facecolor("#FCFBF9"); leg.get_frame().set_linewidth(1.0)
    for t in leg.get_texts(): t.set_color(INK)
 
    fig.tight_layout(rect=[0, 0.02, 1, 0.875])
 
    # within / cross captions just above the top panel (tight, like the earlier version)
    if group_caps and cut is not None:
        pos = axes[0].get_position(); x0, x1, ytop = pos.x0, pos.x1, pos.y1
        xlo, xhi = axb.get_xlim()
        def _fx(dx): return x0 + (dx - xlo)/(xhi - xlo)*(x1 - x0)
        fig.text(_fx((cut-0.5)/2), ytop+0.006, "WITHIN-REPORTER", ha="center", va="bottom",
                 fontsize=9.5, color="#B4ADA3", fontweight="medium")
        fig.text(_fx((cut+(n-1))/2), ytop+0.006, "CROSS-REPORTER", ha="center", va="bottom",
                 fontsize=9.5, color="#B4ADA3", fontweight="medium")
 
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight", facecolor="white"); print("saved", save)
    if show: plt.show()
    return fig
 
# ── (optional) single-metric figure; legend now OUTSIDE so it never covers data ──
def plot_metric(df, metric, save=None, show=False, group_caps=True):
    cfg = METRICS[metric]; div = cfg["divisor"]
    n = len(df); x = np.arange(n)
    off = np.linspace(-0.12, 0.12, len(SERIES))
    fig, ax = plt.subplots(figsize=(13.5, 6.2)); ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    for xi in x: ax.axvline(xi, color=FAINT, lw=0.9, ls=":", zorder=0)
    try:
        cut = np.where(df["group"].str.lower().str.startswith("cross"))[0][0] - 0.5
        ax.axvline(cut, color="#D8D2C8", lw=1.1, ls=(0,(4,3)), zorder=0.5)
    except Exception:
        cut = None
    for (key,label,mk,col,sz,emph), dx in zip(SERIES, off):
        c = f"{metric}_{key}"
        if c not in df.columns: continue
        y = df[c].map(_num).values / div
        ax.scatter(x+dx, y, marker=mk, s=sz, c=col, alpha=0.96, clip_on=False, **_emph_kw(emph))
    ax.set_xticks(x)
    ax.set_xticklabels(_labels(df), fontsize=10.5, color=TICKC, rotation=45,
                       rotation_mode="anchor", ha="right")
    for t in ax.get_xticklabels(): t.set_fontweight(TICKW)
    ax.set_xlim(-0.55, n-0.45); ax.margins(y=0.12)
    if group_caps and cut is not None:
        ymin, ymax = ax.get_ylim(); cy = ymax + (ymax-ymin)*0.02
        ax.text((cut-0.5)/2, cy, "WITHIN-REPORTER", ha="center", va="bottom",
                fontsize=8.5, color="#B4ADA3", fontweight="medium")
        ax.text((cut+(n-1))/2, cy, "CROSS-REPORTER", ha="center", va="bottom",
                fontsize=8.5, color="#B4ADA3", fontweight="medium")
    ax.set_xlabel("Item conversion", fontsize=14, color=INK, labelpad=10, fontweight="bold")
    ax.set_ylabel(cfg["ylabel"], fontsize=14, color=INK, labelpad=10, fontweight="bold")
    ax.tick_params(axis="both", labelsize=10.5, colors=TICKC, length=0)
    for t in ax.get_yticklabels(): t.set_fontweight(TICKW)
    ax.grid(axis="y", ls=":", lw=0.9, color=FAINT, zorder=0)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1.5)
    ax.set_title(cfg["title"]+" across conversions", fontsize=17, color=INK,
                 fontweight="bold", loc="center", pad=30)
    ax.text(0.5, 1.045, "HBI $\\leftrightarrow$ M-PCSI item conversions  ·  70/30 split",
            transform=ax.transAxes, ha="center", fontsize=11, color=GREY,
            family=SERIF, fontstyle="italic")
    handles = [Line2D([0],[0], marker=mk, color="none", markerfacecolor=col,
                      markeredgecolor=("#221F1C" if emph else "white"),
                      markeredgewidth=(1.1 if emph else 0.5),
                      markersize=(14 if emph else 9), label=label)
               for (key,label,mk,col,sz,emph) in SERIES]
    leg = ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                    frameon=False, fontsize=10.5, labelspacing=0.7)
    for t in leg.get_texts(): t.set_color(INK)
    fig.tight_layout()
    if save: fig.savefig(save, dpi=200, bbox_inches="tight", facecolor="white"); print("saved", save)
    if show: plt.show()
    return fig
 
 
df = pd.read_csv("results_acc_70_30_pca_10.csv")
plot_stacked(df, save="./plots/conversion_stacked.png")

#%%

# =============================================================================
# Summary table
# =============================================================================
METRICS = {"acc": "Accuracy", "balacc": "Balanced accuracy",
           "roc": "ROC-AUC",  "f1": "F1"}
MODELS = {
    "1-STS":               "STS_bin",
    "Logistic regression": "logit_raw",
    "STS ensemble (soft)": "ensemble_sts_soft_raw",
    "Ensemble":            "ensemble_raw",
    "SVM (rbf)":           "svm_rbf_raw",
    "Random forest":       "rf_raw",
    "Gradient boosting":   "gb_raw",
}

def mean_of(cell):
    """'67.8 (1.4)' -> 67.8 ; unparseable -> NaN."""
    try:
        return float(str(cell).split()[0])
    except (ValueError, IndexError):
        return np.nan

def summary_table(df, models=MODELS, metrics=METRICS):
    rows = {}
    for label, key in models.items():
        r = {}
        for met, mname in metrics.items():
            vals = df[f"{met}_{key}"].map(mean_of)          # 8 crosswalk means
            r[mname] = f"{vals.mean():.1f} ({vals.std(ddof=1):.1f})"
        rows[label] = r
    return pd.DataFrame(rows).T[list(metrics.values())]

tbl = summary_table(df)
print("Mean (SD) across the 8 crosswalks, x100:\n")
print(tbl.to_string())
tbl.to_csv("model_metric_summary.csv")

# individual panels still available if needed:
# for m in ("acc","balacc","roc","f1"):
#     plot_metric(df, m, save=f"./plots/conversion_{m}.png")

#%%
"""
-----------Not used anymore.. Switched to 1-STS for main code------------------

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
