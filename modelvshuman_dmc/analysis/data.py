import os
import pandas as pd
import numpy as np
from glob import glob
from os.path import join as pjoin
from functools import partial

from .. import constants as c

from pdb import set_trace

def load_human_data(data_dir, expected_subjects=None):
    files = sorted(glob(os.path.join(data_dir, "*subject-*")))
    df = None
    for file in files:
        df_ = pd.read_csv(file)
        df_ = df_.sort_values(by='imagename')
        if 'filename' not in df_.columns:
            df_['filename'] = df_.imagename.apply(lambda img_name: ("_".join(img_name.split("_")[-2:])).removeprefix("00_"))
        if 'is_correct' not in df_.columns:
            df_['is_correct'] = (df_.object_response==df_.category).astype(float)
        df_['condition'] = df_['condition'].astype(str)
        
        df = pd.concat([df, df_])
        
    if expected_subjects is not None:
        num_subjects = len(df.subj.unique())        
        assert num_subjects==expected_subjects, f"Expected num_subjects={expected_subjects}, got {num_subjects}, dataset={dataset}"
        
    return df

def isnan(x):
    return pd.isna(x) or (isinstance(x, float) and np.isnan(x))

def load_model_data(data_dir, model_name):
    files = sorted(glob(os.path.join(data_dir, f"*_{model_name.replace('_', '-')}_*")))
    assert len(files)==1, f"Expected one file, got {files}"
    file = files[0]
    df_ = pd.read_csv(file)
    
    df_ = df_.sort_values(by='imagename')
    if 'filename' not in df_.columns:
        df_['filename'] = df_.imagename.apply(lambda img_name: ("_".join(img_name.split("_")[-2:])).removeprefix("00_"))
    if 'is_correct' not in df_.columns:
        df_['is_correct'] = (df_.object_response==df_.category).astype(float)
    
    df_['condition'] = df_['condition'].apply(lambda x: '0' if isnan(x) else x)
    df_['condition'] = df_['condition'].astype(str)
    
    return df_

def load_models(data_dir, model_names):
    df = pd.concat([load_model_data(data_dir, model_name) for model_name in model_names])
    
    num_subjects = len(df.subj.unique())
    expected_subjects = len(model_names)
    assert num_subjects==expected_subjects, f"Expected num_subjects={expected_subjects}, got {num_subjects}"
        
    return df

def load_modelvsmodel_results(analysis, collection, dataset, data_parent_dir=c.RESULTS_DIR):
    results_dir = pjoin(data_parent_dir, analysis, collection, dataset)
    filename = pjoin(results_dir, f'{collection}_set_{analysis}_{dataset}.csv')
    return pd.read_csv(filename)

def load_modelvsmodel_summary(analysis, collection, dataset, data_parent_dir=c.RESULTS_DIR):
    results_dir = pjoin(data_parent_dir, analysis, collection, dataset)
    filename = pjoin(results_dir, f'{collection}_set_{analysis}_{dataset}_summary.csv')
    return pd.read_csv(filename)