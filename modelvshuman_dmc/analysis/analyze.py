import logging
import time

from fastprogress import master_bar, progress_bar

logger = logging.getLogger(__name__)

class Analyze:
    def __call__(self, subjects, dataset_names, *args, **kwargs):
        """
        Wrapper call to _analyze function.

        Args:
            subjects: ["humans", and any registered model names...]
            dataset_names:
            *args:
            **kwargs:

        Returns:

        """
        logging.info("Results analysis.")
        mb = master_bar(subjects)
        for subj_name in mb:
            for dataset in progress_bar(dataset_names, parent=mb):
                logger.info(f"Running analysis: {subj_name}:{dataset}")
                time.sleep(0.10)
                
        logger.info("Finished analysis.")