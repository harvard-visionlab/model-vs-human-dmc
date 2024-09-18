import pandas as pd
import numpy as np
from natsort import natsorted, index_natsorted
from collections import defaultdict
from scipy import stats
from scipy.stats import pearsonr

from .pairwise_correlations import compute_pairwise_correlations 
from .error_consistency import compute_confidence_interval, compute_error_consistency

from pdb import set_trace

__all__ = [
    'modelvshuman_decision_margin_consistency', 
    'modelvshuman_pairwise_error_consistency',
]

def agg_human_data(human_df, score_col, subject_col='subj', condition_col='condition', item_col='filename'):
    subjects = human_df[subject_col].unique()
    N = len(subjects)
    df_grouped = human_df.groupby([condition_col, item_col])
    counts = df_grouped.count()
    subj_counts = counts[subject_col].unique()
    assert len(subj_counts)==1 and subj_counts[0]==N, "Oops, missing subjects in some conditions"
    
    drop_columns = [col for col in ['Session', 'session', 'trial'] if col in human_df.columns]
    df_avg = df_grouped.mean(numeric_only=True).drop(columns=drop_columns).reset_index()        
    df_avg.rename(columns={score_col: f'human_avg_{score_col}'}, inplace=True)

    return df_avg

def compute_model_vs_human_correlations(model_df, human_df, score_col, human_score_col='is_correct', subject_col='subj', condition_col='condition', item_col='filename'):
    '''correlate scores across all pairs of subjects separately for each condition'''
    
    # aggregate the human data
    human_df_avg = agg_human_data(human_df, human_score_col, subject_col=subject_col, condition_col=condition_col, item_col=item_col)
        
    subjects = model_df[subject_col].unique()
    conditions = natsorted(model_df[condition_col].unique())
    
    groupby = [condition_col, item_col]
    results = defaultdict(list)
    correlations = defaultdict(list)
    for model_num, model_name in enumerate(subjects):
        model_df_grouped = model_df[model_df[subject_col]==model_name].groupby(groupby)[score_col].mean().reset_index()
        model_df_grouped.rename(columns={score_col: f'model_{score_col}'}, inplace=True)
        
        merged_df = pd.merge(human_df_avg, model_df_grouped, on=groupby, how='outer')
        
        try:
            assert len(merged_df)==len(human_df_avg), "Failed to align model/human datasets using dataframe merge"
            assert len(merged_df)==len(model_df_grouped), "Failed to align model/human datasets using dataframe merge"        
        except:
            set_trace()
            
        for condition in conditions:
            cond_df = merged_df[merged_df[condition_col] == condition]
            r = pearsonr(cond_df[f'human_avg_{human_score_col}'], cond_df[f'model_{score_col}'])[0]
            correlations[condition].append(r)
            results['model_num'].append(model_num)
            results['model_name'].append(model_name)
            results[condition_col].append(condition)
            results['pearsonr'].append(r)
    
    summary = dict()
    for condition in conditions:
        scores = correlations[condition]
        avg_correlation = np.mean(scores)
        sem_correlation = stats.sem(scores)
        
        # Compute the 95% confidence interval
        confidence_interval = stats.t.interval(0.95, len(scores) - 1, 
                                               loc=avg_correlation, scale=sem_correlation)
        
        summary[condition] = dict(
            avg_correlation=avg_correlation, 
            sem_correlation=sem_correlation,
            avg_corr_lower_ci=confidence_interval[0],
            avg_corr_upper_ci=confidence_interval[1],
            min_correlation=np.min(scores),
            max_correlation=np.max(scores),
        )
    
    df = pd.DataFrame(summary)
    df_summary = df.T
    df_summary = df_summary.reset_index()
    df_summary = df_summary.rename(columns={'index': condition_col})

    return dict(results), df_summary

def modelvshuman_decision_margin_consistency(model_df, human_df, condition_col='condition', **kwargs):
    '''compare models' accuracy for each item/image'''
    results, _ = compute_model_vs_human_correlations(model_df, human_df, score_col='decision_margin', condition_col=condition_col, **kwargs)
    results['decision_margin_consistency'] = results.pop('pearsonr')
    
    return results, None

def modelvshuman_pairwise_error_consistency(model_df, human_df, condition_col='condition', **kwargs):
    results, _ = compute_error_consistency(model_df, human_df, condition_col=condition_col, **kwargs)    
    
    # compute summary
    results_df = pd.DataFrame(results)
    groupby = [condition_col, 'sub1']
    drop_cols = ['pair_num', 'sub1_pct_correct', 'sub2_pct_correct']
    df_avg = results_df.groupby(by=groupby).mean(numeric_only=True).reset_index().drop(columns=drop_cols)    
    df_avg = df_avg.iloc[index_natsorted(df_avg[condition_col])]
        
    # rename columns
    columns_to_rename = ['expected_consistency', 'observed_consistency', 'error_consistency']
    renaming_dict = {col: f'{col}_avg' for col in columns_to_rename}
    df_avg = df_avg.rename(columns=renaming_dict)
    
    # Compute confidence intervals for 'avg_error_consistency'
    ci_df = results_df.groupby(by=groupby)['error_consistency'].apply(
        lambda x: pd.Series(compute_confidence_interval(x), index=['error_consistency_lower_ci', 'error_consistency_upper_ci'])
    ).reset_index()    

    # Pivot the DataFrame using the groupby columns and reset the index
    num_levels = len(groupby)
    ci_df_wide = ci_df.pivot(index=groupby, columns=f'level_{num_levels}', values='error_consistency').reset_index()
    ci_df_wide.columns.name = None
    ci_df_wide = ci_df_wide.iloc[index_natsorted(ci_df_wide[condition_col])]
    
    df_summary = pd.merge(df_avg, ci_df_wide, on=groupby, how='left')
    
    return results, df_summary   