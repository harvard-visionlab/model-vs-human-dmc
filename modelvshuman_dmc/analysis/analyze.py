import os
import time
import logging
import pandas as pd
from collections import defaultdict
from itertools import product
from tqdm import tqdm
from os.path import join as pjoin
from pathlib import Path

from .. import constants as c
from . import data
from . import analyses

from pdb import set_trace

logger = logging.getLogger(__name__)

class ResultsRecorder:
    def __init__(self, subj, dataset, analysis, data_parent_dir=c.RESULTS_DIR):
        self.subj = subj
        self.dataset = dataset
        self.analysis = analysis
        self.raw_data_dir = pjoin(c.RAW_DATA_DIR, dataset)
        self.results_dir = pjoin(data_parent_dir, analysis, dataset)
        Path(self.raw_data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        self.filename = pjoin(self.results_dir, f'{analysis}_{dataset}.csv')
        self.file_exists = os.path.exists(self.filename)
        self.data = defaultdict(list)
    
    def update(self, record):
        for k,v in record.items(): 
            self.data[k].append(v)
        
    def as_dataframe(self):
        df = pd.DataFrame(self.data)
        df.insert(0, 'subj', self.subj)
        df.insert(1, 'dataset', self.dataset)
        df.insert(2, 'analysis', self.analysis)
        return df
    
    def save(self):
        df = self.as_dataframe()
        df.to_csv(self.filename, index=False)
        
    def save_summary(self, df):
        filename = self.filename.replace(".csv", "_summary.csv")
        df.to_csv(filename, index=False)
        
class Analyze:
    
    def _humanvshuman_analysis(self, model_name, dataset, analysis, force_recompute=False):
        # get analysis function
        analyzer = analyses.__dict__[analysis]
        
        # setup the results recorder
        results = ResultsRecorder(analysis=analysis, dataset=dataset, subj='humanvshuman')
        if results.file_exists and force_recompute==False:
            logging.info(f"File exists, skipping: {results.filename}")
            print(f"File exists, skipping: {results.filename}")
            
        # load the human data
        df = data.load_human_data(results.raw_data_dir)
        num_subjects = len(df.subj.unique())
        expected_subjects = c.EXPECTED_SUBJECTS.get(dataset, 4)
        assert num_subjects==expected_subjects, f"Expected num_subjects={expected_subjects}, got {num_subjects}, dataset={dataset}"
        
        # run the split halves analysis
        results.data, summary = analyzer(df)
        
        # save the results
        results.save()
        results.save_summary(summary)
        
    def _modelvsmodel_analysis(self, model_name, dataset, analysis, force_recompute=False):
        # setup the results recorder
        results = ResultsRecorder(model_name=model_name, dataset=dataset, analysis=analysis)
    
    def _modelvshuman_analysis(self, model_name, dataset, analysis, force_recompute=False):
        # setup the results recorder
        results = ResultsRecorder(model_name=model_name, dataset=dataset, analysis=analysis)
    
    def _get_analysis_runner(self, analysis):
        if analysis.startswith("humanvshuman"):
            return self._humanvshuman_analysis
        elif analysis.startswith("modelvsmodel"):
            return self._modelvsmodel_analysis
        elif analysis.startswith("modelvshuman"):
            return self._modelvshuman_analysis        
        else:
            raise NameError(f"Unsupported analisys {analysis}")
            
    def __call__(self, models, dataset_names, analyses, *args, force_recompute=False, **kwargs):
        """
        Wrapper call to _analyze function.

        Args:
            subjects: ["humans", and any registered model names...]
            dataset_names:
            *args:
            **kwargs:

        Returns:

        """
        logging.info("Running analyses...")
        
        # Generate all possible combinations of (model, dataset, analysis)
        tasks = product(models, dataset_names, analyses)
        
        # Initialize the tqdm progress bar
        progress_bar = tqdm(tasks, total=len(models)*len(dataset_names)*len(analyses), desc='Running Analyses')

        for model, dataset, analysis in progress_bar:
            # Update the progress bar with the current task
            progress_bar.set_postfix({'Model': model, 'Dataset': dataset, 'Analysis': analysis})            
            logging_info = f"Running Human-vs-Human Analysis on model {model} and dataset {dataset}"
            logger.info(logging_info)
            print(logging_info)
            
            # run the analysis
            analysis_runner = self._get_analysis_runner(analysis)                                    
            analysis_runner(model, dataset, analysis)
                
        logger.info("Finished analysis.")
        print("Finished analyses.")