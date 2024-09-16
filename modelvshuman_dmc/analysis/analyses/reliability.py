import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import stats
from scipy.stats import pearsonr, sem
from itertools import combinations
from natsort import natsorted

def get_split_halves(N):
    '''get all possible, unique splits of N subjects into two groups'''
    subjects = list(range(0,N))
    splits = []
    for subsetA in combinations(subjects, N//2):
        subsetA = list(subsetA)
        subsetB = list(np.setdiff1d(subjects, subsetA))
        assert len(np.setdiff1d(subsetA,subsetB)) == len(subsetA), "oops"
        assert len(np.setdiff1d(subsetB,subsetA)) == len(subsetB), "oops"
        assert (len(subsetA) + len(subsetB)) == N, f"oops, total should be {N}"
        splits.append((subsetA,subsetB))
    
    return splits[0:len(splits)//2] if N%2==0 else splits

def compute_splithalf_reliability(df):
    '''compute split half reliability
        - for all possible splits of the subjects into two groups (groupA and groupB)
        - for each group (separately) compute average accuracy for each image across subjects in that group,
        - then correlate the item-scores between groupA and groupB.
        - the avg correlation across all splits is used to estimate the split-half 
          reliability
        - the spearman brown adjustment is used to estimate the reliability
          of the full dataset (aka the noise ceiling).
    '''
    # compute split half reliability
    subjects = df.subj.unique()
    conditions = natsorted(df.condition.unique())
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
    
    df = pd.DataFrame(summary)
    df_summary = df.T
    df_summary = df_summary.reset_index()
    df_summary = df_summary.rename(columns={'index': 'condition'})

    return results, df_summary