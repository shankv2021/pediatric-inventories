# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 20:41:35 2026

@author: vshas
"""

import numpy as np
import pandas as pd

def build_df_ov(df, max_missing_items=1):
    """
    Build the overlap analysis sample.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw data as read from the Excel sheet. Not modified in place.
    max_missing_items : int, default 1
        A questionnaire block counts as "usable" if it has at most this many
        missing items.

    Returns
    -------
    pandas.DataFrame
        `df_ov`: unique subjects (by subject_ID) with usable HBI and M-PCSI
        blocks, including the added `female` indicator and the four `*_sum`
        total-score columns.
    """
     
    # ----------------------------------------------------------------------------- 
    # Build a single Female indicator from sex + gender
    #   sex 0=Male / 1=Female ; gender 0=Male / 1=Female
    #   sex   carries {'Male','Female',0,1}
    #   gender carries {'M','F',0,1}
    # ----------------------------------------------------------------------------- 
    MALE_TOKENS   = {"male", "m", "0", "0.0"}
    FEMALE_TOKENS = {"female", "f", "1", "1.0"}
     
    def to_female(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if s in MALE_TOKENS:
            return 0.0
        if s in FEMALE_TOKENS:
            return 1.0
        return np.nan
     
    df["female"] = df["sex"].map(to_female).combine_first(df["gender"].map(to_female))
     
    # ----------------------------------------------------------------------------- 
    # Item lists and total scores (each sum uses its OWN items)
    # ----------------------------------------------------------------------------- 
    HBI_c_items  = [f"HBIc{i:02d}"  for i in range(1, 21)]   # HBI child      (20 items)
    HBI_p_items  = [f"HBIp{i:02d}"  for i in range(1, 21)]   # HBI parent     (20 items)
    MPCS_c_items = [f"MPCSc{i:02d}" for i in range(1, 16)]   # M-PCSI child   (15 items)
    MPCS_p_items = [f"MPCSp{i:02d}" for i in range(1, 16)]   # M-PCSI parent  (15 items)
     
    ITEM_GROUPS = {
        "HBIc_sum":  HBI_c_items,
        "HBIp_sum":  HBI_p_items,
        "MPCSc_sum": MPCS_c_items,
        "MPCSp_sum": MPCS_p_items,
    }
    for name, cols in ITEM_GROUPS.items():
        df[name] = df[cols].sum(axis=1)   # .sum() skips NaN, so <=1 missing item is fine
     
    # ----------------------------------------------------------------------------- 
    # "Usable" masks + overlap sample (subjects with usable HBI *and* M-PCSI)
    # ----------------------------------------------------------------------------- 
    def few_missing(cols):
        return df[cols].isnull().sum(axis=1) <= max_missing_items
     
    hbi_ok  = few_missing(HBI_p_items)  & few_missing(HBI_c_items)
    mpcs_ok = few_missing(MPCS_p_items) & few_missing(MPCS_c_items)
     
    df_ov = df[hbi_ok & mpcs_ok].drop_duplicates("subject_ID", keep="first").copy()
    
    return df_ov