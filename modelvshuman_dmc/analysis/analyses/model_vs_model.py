import pandas as pd
from natsort import index_natsorted

# split-halves for models only makes sense if you can average over models (e.g., alexnets with different seeds)
# from .reliability import compute_splithalf_reliability as modelvsmodel_splithalves
from .pairwise_correlations import compute_pairwise_correlations 
from .error_consistency import compute_confidence_interval, compute_error_consistency

from pdb import set_trace

__all__ = [
    'modelvsmodel_pairwise_decision_margin_consistency', 
    'modelvsmodel_pairwise_error_consistency',
]

def modelvsmodel_pairwise_decision_margin_consistency(human_df, model_df, condition_col='condition', **kwargs):
    '''compare models' accuracy for each item/image'''
    results, summary = compute_pairwise_correlations(*args, condition_col=condition_col, 
                                                     score='decision_margin', **kwargs)
    results['decision_margin_consistency'] = results.pop('pearsonr')
    
    return results, summary

def modelvsmodel_pairwise_error_consistency(*args, condition_col='condition', **kwargs):
    results, _ = compute_error_consistency(*args, condition_col=condition_col, **kwargs)
    
    # compute summary
    results_df = pd.DataFrame(results)
    groupby = [condition_col]
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
    ci_df_wide = ci_df.pivot(index=groupby, columns='level_1', values='error_consistency').reset_index()
    ci_df_wide.columns.name = None
    ci_df_wide = ci_df_wide.iloc[index_natsorted(ci_df_wide[condition_col])]
    
    df_summary = pd.merge(df_avg, ci_df_wide, on=groupby, how='left')
    
    return results, df_summary    