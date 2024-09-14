import time
import logging
import pandas as pd
from collections import defaultdict
from itertools import product
from tqdm import tqdm
from os.path import join as pjoin

from .. import constants as c
from . import data
from .analyses.human_vs_human import human_vs_human_splithalves

from pdb import set_trace

logger = logging.getLogger(__name__)

class ResultsRecorder:
    def __init__(self, model_name, dataset, analysis, data_parent_dir=c.RESULTS_DIR):

        self.model_name = model_name
        self.dataset = dataset
        self.analysis = analysis
        self.raw_data_dir = pjoin(c.RAW_DATA_DIR, dataset)
        self.results_dir = pjoin(data_parent_dir, dataset)
        self.filename = pjoin(self.results_dir, f'{dataset}_{model_name}_{analysis}.csv')
        self.data = defaultdict(list)
    
    def update(self, record):
        for k,v in record.items(): 
            self.data[k].append(v)
        
    def as_dataframe(self):
        df = pd.DataFrame(self.data)
        df.insert(0, 'analysis', self.analysis)
        df.insert(1, 'model_name', self.model_name)
        df.insert(2, 'dataset', self.dataset)        
        return df
    
    def save(self):
        df = self.as_dataframe()
        df.to_csv(self.filename, index=False)
        
class Analyze:
    def _human_vs_human_analysis(self, model_name, dataset, analysis):
        # setup the results recorder
        results = ResultsRecorder(model_name=model_name, dataset=dataset, analysis=analysis)
        
        # load the human data
        df = data.load_human_data(results.raw_data_dir)
        num_subjects = len(df.subj.unique())
        assert num_subjects==4, f"Expected num_subjects=4, got {num_subjects}"
        
        # run the split halves analysis
        results.data, summary = human_vs_human_splithalves(df)
        set_trace()
        
    def _model_vs_model_analysis(self, model_name, dataset, analysis):
        # setup the results recorder
        results = ResultsRecorder(model_name=model_name, dataset=dataset, analysis=analysis)
    
    def _model_vs_human_analysis(self, model_name, dataset, analysis):
        # setup the results recorder
        results = ResultsRecorder(model_name=model_name, dataset=dataset, analysis=analysis)
    
    def _get_analyzer(self, analysis):
        if analysis == "human_vs_human":
            return self._human_vs_human_analysis
        elif analysis == "model_vs_model":
            return self._model_vs_model_analysis
        elif analysis == "model_vs_human":
            return self._model_vs_human_analysis        
        else:
            raise NameError(f"Unsupported analisys {analysis}")
            
    def __call__(self, models, dataset_names, analyses, *args, **kwargs):
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
            analyzer = self._get_analyzer(analysis)                                    
            analyzer(model, dataset, analysis)
                
        logger.info("Finished analysis.")
        print("Finished analyses.")