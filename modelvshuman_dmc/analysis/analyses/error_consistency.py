import os
import pandas as pd
import numpy as np
from glob import glob
from itertools import combinations
from scipy.stats import pearsonr
from collections import defaultdict
from tqdm import tqdm

def error_consistency(expected_consistency, observed_consistency):
        """Return error consistency as measured by Cohen's kappa."""

        assert expected_consistency >= 0.0
        assert expected_consistency <= 1.0
        assert observed_consistency >= 0.0
        assert observed_consistency <= 1.0

        if observed_consistency == 1.0:
            return 1.0
        else:
            return (observed_consistency - expected_consistency) / (1.0 - expected_consistency)
    
def expected_consistency(df1, df2):
    p1 = df1.is_correct.mean()
    p2 = df2.is_correct.mean()
    expected_consistency = p1 * p2 + (1 - p1) * (1 - p2)
    
    return expected_consistency, p1, p2

def observed_consistency(df1, df2):
    return (df1.is_correct == df2.is_correct).sum() / len(df1)

def compute_error_consistency(df1, df2):
    expected_con, p1, p2 = expected_consistency(df1, df2)
    observed_con = observed_consistency(df1, df2)
    error_con = error_consistency(expected_con, observed_con)
    return expected_con, observed_con, error_con

def compute_human_vs_model_error_consistency(human, model):
    human_subjects = human.subj.unique()
    human_cond = human.condition.unique()
    model_subjects = model.subj.unique()
    model_cond = model.condition.unique()
    assert (human_cond == model_cond).all(), "Human and Model data must contain the same conditions"
    
    results = defaultdict(list)
    for human_subj in human_subjects:        
        for model_subj in model_subjects:
            for condition in conditions:
                df1 = human[(human.subj == human_subj) & (human.condition==condition)].sort_values(by='filename').reset_index(drop=True)
                df2 = model[(model.subj == model_subj) & (model.condition==condition)].sort_values(by='filename').reset_index(drop=True)
                expected_con, p1, p2 = expected_consistency(df1, df2)
                observed_con = observed_consistency(df1, df2)
                error_con = error_consistency(expected_con, observed_con)
                
                results['condition'].append(condition)
                results['human_subj'].append(human_subj)
                results['model_subj'].append(model_subj)
                
                results['human_pct_correct'].append(p1)
                results['model_pct_correct'].append(p2)
                
                results['expected_consistency'].append(expected_con)
                results['observed_consistency'].append(observed_con)
                results['error_consistency'].append(error_con)
    
    return pd.DataFrame(results)