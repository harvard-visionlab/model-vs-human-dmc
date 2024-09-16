import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import stats
from scipy.stats import pearsonr, sem
from itertools import combinations
from natsort import natsorted

def compute_pairwise_correlations(df, subject='subj', condition='condition', item='filename', score='is_correct'):
    '''correlate scores across all pairs of subjects separately for each condition'''

    subjects = df[subject].unique()
    conditions = natsorted(df[condition].unique())
    pairs = list(combinations(subjects, 2))
    
    groupby = [condition, item]
    results = defaultdict(list)
    correlations = defaultdict(list)
    for pair_num, (subA,subB) in enumerate(pairs):
        dfA = df[df[subject]==subA]
        dfB = df[df[subject]==subB]

        grouped_A = dfA.groupby(groupby)[score].mean().reset_index()
        grouped_A.rename(columns={score: 'mean_A'}, inplace=True)

        grouped_B = dfB.groupby(groupby)[score].mean().reset_index()
        grouped_B.rename(columns={score: 'mean_B'}, inplace=True)

        merged_df = pd.merge(grouped_A, grouped_B, on=groupby, how='outer')

        for condition in conditions:
            cond_df = merged_df[merged_df['condition'] == condition]
            r = pearsonr(cond_df.mean_A, cond_df.mean_B)[0]
            correlations[condition].append(r)
            results['pair_num'].append(pair_num)
            results['subject_A'].append(subA)
            results['subject_B'].append(subB)
            results['condition'].append(condition)
            results['pearsonr'].append(r)
    
    summary = dict()
    for condition in conditions:
        scores = correlations[condition]
        avg_correlation = np.mean(scores)
        sem_correlation = sem(scores)
        
        # Compute the 95% confidence interval
        confidence_interval = stats.t.interval(0.95, len(scores) - 1, 
                                               loc=avg_correlation, scale=sem_correlation)
        
        summary[condition] = dict(
            avg_correlation=avg_correlation, 
            sem_correlation=sem_correlation,
            avg_corr_lower_ci=confidence_interval[0],
            avg_corr_upper_ci=confidence_interval[1],
        )
    
    df = pd.DataFrame(summary)
    df_summary = df.T
    df_summary = df_summary.reset_index()
    df_summary = df_summary.rename(columns={'index': 'condition'})

    return results, df_summary