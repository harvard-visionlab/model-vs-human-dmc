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
from .analyses import model_vs_model

from pdb import set_trace

logger = logging.getLogger(__name__)

class ResultsRecorder:
    def __init__(self, dataset, analysis, collection=None, data_parent_dir=c.RESULTS_DIR):        
        self.dataset = dataset
        self.analysis = analysis
        self.subject_group = analysis.split("_")[0]
        self.collection = collection # modelvs... analyses can be done for different collections of models
        self.raw_data_dir = pjoin(c.RAW_DATA_DIR, dataset)   
        
        if collection is not None:
            self.results_dir = pjoin(data_parent_dir, analysis, collection, dataset)
            self.filename = pjoin(self.results_dir, f'{collection}_set_{analysis}_{dataset}.csv')
        else:
            self.results_dir = pjoin(data_parent_dir, analysis, dataset)
            self.filename = pjoin(self.results_dir, f'{analysis}_{dataset}.csv')
            
        Path(self.raw_data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        self.file_exists = os.path.exists(self.filename)
        self.data = defaultdict(list)
    
    def update(self, record):
        for k,v in record.items(): 
            self.data[k].append(v)
    
    def prepend_columns(self, df):
        df.insert(0, 'dataset', self.dataset)
        df.insert(1, 'analysis', self.analysis)
        df.insert(2, 'subject_group', self.subject_group)
        if self.collection is not None:
            df.insert(3, 'collection', self.collection)            
        return df
    
    def as_dataframe(self):
        df = pd.DataFrame(self.data) 
        df = self.prepend_columns(df)
        return df
    
    def save(self):
        df = self.as_dataframe()
        df.to_csv(self.filename, index=False)
        
    def save_summary(self, df):
        df = self.prepend_columns(df)
        filename = self.filename.replace(".csv", "_summary.csv")
        df.to_csv(filename, index=False)
        
class Analyze:
    
    def _humanvshuman_analysis(self, dataset, analysis, *args, force_recompute=False, **kwargs):
        # get analysis function
        analyzer = analyses.__dict__[analysis]
        
        # setup the results recorder
        results = ResultsRecorder(analysis=analysis, dataset=dataset)
        if results.file_exists and force_recompute==False:
            logging.info(f"File exists, skipping: {results.filename}")
            print(f"File exists, skipping: {results.filename}")
            
        # load the human data
        df = data.load_human_data(results.raw_data_dir, expected_subjects=c.EXPECTED_SUBJECTS.get(dataset, 4))

        # run the analysis
        results.data, summary = analyzer(df)
        
        # save the results
        results.save()
        
        # save summary
        if summary is not None:
            results.save_summary(summary)
        
    def _modelvsmodel_analysis(self, dataset, analysis, models, model_collection, force_recompute=False, **kwargs):
        
        # get analysis function
        analyzer = model_vs_model.__dict__[analysis]
        
        # record the results separately for each modelA-modelB pair
        results = ResultsRecorder(analysis=analysis, dataset=dataset, collection=model_collection)
        if results.file_exists and force_recompute==False:
            logging.info(f"File exists, skipping: {results.filename}")
            print(f"File exists, skipping: {results.filename}")
        
        # load the models
        raw_data_dir = pjoin(c.RAW_DATA_DIR, dataset)
        df = data.load_models(raw_data_dir, models)        
        
        # run the analysis
        results.data, summary = analyzer(df)
        
        # save the results
        results.save()
        
        # save summary
        if summary is not None:
            results.save_summary(summary)
            
    def _modelvshuman_analysis(self, dataset, analysis, models, model_collection, force_recompute=False, **kwargs):

        # get analysis function
        analyzer = analyses.__dict__[analysis]
        
        # setup the results recorder
        results = ResultsRecorder(analysis=analysis, dataset=dataset, collection=model_collection)
        if results.file_exists and force_recompute==False:
            logging.info(f"File exists, skipping: {results.filename}")
            print(f"File exists, skipping: {results.filename}")
        
        # load the human data
        human_df = data.load_human_data(results.raw_data_dir, expected_subjects=c.EXPECTED_SUBJECTS.get(dataset, 4))
        
        # load the models
        raw_data_dir = pjoin(c.RAW_DATA_DIR, dataset)
        model_df = data.load_models(raw_data_dir, models) 
        
        # run the analysis
        results.data, summary = analyzer(model_df=model_df, human_df=human_df)
        
        # save the results
        results.save()
        
        # save summary
        if summary is not None:
            results.save_summary(summary)
    
    def _get_analysis_runner(self, analysis):
        if analysis.startswith("humanvshuman"):
            return self._humanvshuman_analysis
        elif analysis.startswith("modelvsmodel"):
            return self._modelvsmodel_analysis
        elif analysis.startswith("modelvshuman"):
            return self._modelvshuman_analysis        
        else:
            raise NameError(f"Unsupported analysis {analysis}")
            
    def __call__(self, models, dataset_names, analyses, *args, **kwargs):
        """
        Wrapper call to _analyze function.

        Args:
            models: [list of registered model names]
            dataset_names: [list of datasets]
            analyses: [list of analyses]
            *args:
            **kwargs:

        Returns:

        """
        logging.info("Running analyses...")
        
        # Generate all possible combinations of (dataset, analysis)
        tasks = list(product(dataset_names, analyses))
        
        # Initialize the tqdm progress bar
        progress_bar = tqdm(tasks, total=len(tasks), desc='Running Analyses')
        
        for dataset, analysis in progress_bar:
            progress_bar.set_postfix({'Dataset': dataset, 'Analysis': analysis})
            
            # different analysis runners for humanvshuman, modelvsmodel, and modelvshuman
            analysis_runner = self._get_analysis_runner(analysis)
            
            # run the analysis
            analysis_runner(dataset, analysis, models, *args, **kwargs)
                
        logger.info("Finished analysis.")
        print("Finished analyses.")