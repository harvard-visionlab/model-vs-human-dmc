import os
import pandas as pd
from glob import glob

def load_human_data(data_dir):
    files = sorted(glob(os.path.join(data_dir, "*subject-*")))
    df = None
    for file in files:
        df_ = pd.read_csv(file)
        df_ = df_.sort_values(by='imagename')
        if 'filename' not in df_.columns:
            df_['filename'] = df_.imagename.apply(lambda x: "_".join(x.split("_")[-2:]))
        if 'is_correct' not in df_.columns:
            df_['is_correct'] = (df_.object_response==df_.category).astype(float)
        df = pd.concat([df, df_])
        
    return df

def load_model_data(data_dir, model_name):
    files = sorted(glob(os.path.join(data_dir, f"*{model_name}*")))
    assert len(files)==1, f"Expected one file, got {files}"
    file = files[0]
    df_ = pd.read_csv(file)
    df_ = df_.sort_values(by='imagename')
    if 'filename' not in df_.columns:        
        df_['filename'] = df_.imagename.apply(lambda x: "_".join(x.split("_")[-2:]))
    if 'is_correct' not in df_.columns:
        df_['is_correct'] = (df_.object_response==df_.category).astype(float)
        
    return df_