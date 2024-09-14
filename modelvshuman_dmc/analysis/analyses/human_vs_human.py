import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import stats
from scipy.stats import pearsonr, sem

from ..split_halves import get_split_halves

__all__ = ['human_vs_human_splithalves']

def human_vs_human_splithalves(df):
    '''compute split half reliability
        - for all possible splits of the subjects into two groups (groupA and groupB)
        - compute average accuracy for each image across subjects (separately for each group),
          the correlate scores across items between groupA and groupB.
        - the avg correlation across all splits is used to estimate the split-half 
        reliability
        - the spearman brown adjustment is used to estimate the reliability
          of the full dataset (aka the noise ceiling).
    '''
    # compute split half reliability
    subjects = df.subj.unique()
    conditions = df.condition.unique()
    splits = get_split_halves(len(subjects))
    
    groupby = ['condition', 'filename']
    results = defaultdict(list)
    correlations = defaultdict(list)
    for split_num, (splitA,splitB) in enumerate(splits):
        subA = subjects[splitA]
        subB = subjects[splitB]
        dfA = df[df.subj.isin(subA)]
        dfB = df[df.subj.isin(subB)]

        grouped_A = dfA.groupby(groupby)['is_correct'].mean().reset_index()
        grouped_A.rename(columns={'is_correct': 'mean_correct_A'}, inplace=True)

        grouped_B = dfB.groupby(groupby)['is_correct'].mean().reset_index()
        grouped_B.rename(columns={'is_correct': 'mean_correct_B'}, inplace=True)

        merged_df = pd.merge(grouped_A, grouped_B, on=groupby, how='outer')

        for condition in conditions:
            cond_df = merged_df[merged_df.condition == condition]
            r = pearsonr(cond_df.mean_correct_A, cond_df.mean_correct_B)[0]
            correlations[condition].append(r)
            results['split_num'].append(split_num)
            results['splitA'].append(subA)
            results['splitB'].append(subB)
            results['condition'].append(condition)
            results['pearsonr'].append(r)
    
    summary = dict()
    for condition in conditions:
        scores = correlations[condition]
        adjusted_correlations = [(2 * r) / (1 + r) for r in scores]
        avg_split_half_corr = np.mean(scores) # <-- really should fisherz these first
        
        avg_adj_correlation = np.mean(adjusted_correlations)
        sem_adj_correlation = sem(adjusted_correlations)
        
        # Compute the 95% confidence interval
        confidence_interval = stats.t.interval(0.95, len(adjusted_correlations) - 1, 
                                               loc=avg_adj_correlation, scale=sem_adj_correlation)
        
        summary[condition] = dict(
            avg_split_half_corr=avg_split_half_corr, 
            adj_corr_mean=avg_adj_correlation,
            adj_corr_sem=sem_adj_correlation,
            adj_corr_lower_ci=confidence_interval[0],
            adj_corr_upper_ci=confidence_interval[1],
        )
        
    return results, summary