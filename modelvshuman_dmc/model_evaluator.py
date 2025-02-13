import copy
import datetime
import logging
import os
import numpy as np

import torch
from tqdm import tqdm
from pdb import set_trace

from .helper.decision_margin_distance import compute_decision_margin_distance
from .datasets import ToTensorflow
from .evaluation import evaluate as e
from .utils import load_dataset, load_model

from .models.wrappers.pytorch import PytorchModel
from .models.wrappers.tensorflow import TensorflowModel

logger = logging.getLogger(__name__)
MAX_NUM_MODELS_IN_CACHE = 3


def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModelEvaluator:

    def _pytorch_evaluator(self, model_name, model, dataset, *args, **kwargs):
        """
        Evaluate Model on the given dataset and return the accuracy.
        Args:
            model_name:
            model:
            dataset:
            *args:
            **kwargs:
        """

        logging_info = f"Evaluating model {model_name} on dataset {dataset.name} using Pytorch Evaluator"
        logger.info(logging_info)
        print(logging_info)
        for metric in dataset.metrics:
            metric.reset()
        with torch.no_grad():
            result_writer = e.ResultPrinter(model_name=model_name,
                                            dataset=dataset)

            for images, target, paths in tqdm(dataset.loader):
                images = images.to(device())
                logits = model.forward_batch(images)                
                if isinstance(target, torch.Tensor):
                    batch_targets = model.to_numpy(target)
                else:
                    batch_targets = target
                # softmax_output = model.softmax(logits) # aggregate logits, then softmax
                predictions, sorted_probs, sorted_logits = dataset.decision_mapping(logits, model.softmax)
                
                # get the response (logit) to the target
                target_mask = predictions == np.array(batch_targets)[:, np.newaxis]                                
                target_act = sorted_logits[target_mask]
                
                # get the max response (logit) among non-targets
                non_targets = ~target_mask
                non_target_act = np.where(non_targets, sorted_logits, np.nan)
                max_nontarget_act = np.nanmax(non_target_act, axis=1)
                decision_margin = compute_decision_margin_distance(target_act, max_nontarget_act)                
        
                for metric in dataset.metrics:
                    metric.update(predictions,
                                  batch_targets,
                                  paths)
                if kwargs["print_predictions"]:
                    result_writer.print_batch_to_csv(object_response=predictions,
                                                     batch_targets=batch_targets,
                                                     paths=paths,
                                                     target_act=target_act,
                                                     max_nontarget_act=max_nontarget_act, 
                                                     decision_margin=decision_margin)

    def _tensorflow_evaluator(self, model_name, model, dataset, *args, **kwargs):
        """
        Evaluate Model on the given dataset and return the accuracy.
        Args:
            model_name:
            model:
            dataset:
            *args:
            **kwargs:

        Returns:
            accuracy
        """

        logging_info = f"Evaluation model {model_name} on dataset {dataset.name} using Tensorflow Evaluator"
        logger.info(logging_info)
        print(logging_info)
        result_writer = e.ResultPrinter(model_name=model_name,
                                        dataset=dataset)
        for metric in dataset.metrics:
            metric.reset()
        for images, target, paths in tqdm(dataset.loader):
            logits = model.forward_batch(images)
            softmax_output = model.softmax(logits)
            predictions, probs = dataset.decision_mapping(softmax_output)
            for metric in dataset.metrics:
                metric.update(predictions,
                              target,
                              paths)
            if kwargs["print_predictions"]:
                result_writer.print_batch_to_csv(object_response=predictions,
                                                 batch_targets=target,
                                                 paths=paths)

    def _get_datasets(self, dataset_names, *args, **kwargs):
        dataset_list = []
        for dataset in dataset_names:
            dataset = load_dataset(dataset, *args, **kwargs)
            dataset_list.append(dataset)
        return dataset_list

    def _to_tensorflow(self, datasets):
        datasets = copy.deepcopy(datasets)
        new_datasets = []
        for dataset in datasets:
            dataset.loader = ToTensorflow(dataset.loader)
            new_datasets.append(dataset)
        return new_datasets

    def _get_evaluator(self, framework):
        if framework == "tensorflow":
            return self._tensorflow_evaluator
        elif framework == 'pytorch':
            return self._pytorch_evaluator
        else:
            raise NameError("Unsupported evaluator")

    def _remove_model_from_cache(self, framework, model_name):

        def _format_name(name):
            return name.lower().replace("-", "_")

        try:
            if framework == "pytorch":
                cachedir = "/root/.cache/torch/checkpoints/"
                downloaded_models = os.listdir(cachedir)
                for dm in downloaded_models:
                    if _format_name(dm).startswith(_format_name(model_name)):
                        os.remove(os.path.join(cachedir, dm))
        except:
            pass

    def __call__(self, models, dataset_names, *args, **kwargs):
        """
        Wrapper call to _evaluate function.

        Args:
            models:
            dataset_names:
            *args:
            **kwargs:

        Returns:

        """
        logging.info("Model evaluation.")
        _datasets = self._get_datasets(dataset_names, *args, **kwargs)
        model_names = []
        for model_name in models:
            datasets = _datasets
            if isinstance(model_name, str):                
                model, framework = load_model(model_name, *args)
            elif callable(model_name):
                model = model_name(model_name, *args)
                model_name = model.model_name
                if isinstance(model, PytorchModel):
                    framework = "pytorch"
                elif isinstance(model, TensorflowModel):
                    framework = "tensorflow"
                else:
                    raise ValueError(f"Unrecognized model type: {model}")
            model_names.append(model_name)
            
            evaluator = self._get_evaluator(framework)
            if framework == 'tensorflow':
                datasets = self._to_tensorflow(datasets)
            logger.info(f"Loaded model: {model_name}")
            for dataset in datasets:
                # start time
                time_a = datetime.datetime.now()
                evaluator(model_name, model, dataset, *args, **kwargs)
                for metric in dataset.metrics:
                    logger.info(str(metric))
                    print(metric)

                # end time
                time_b = datetime.datetime.now()
                c = time_b - time_a

                if kwargs["print_predictions"]:
                    # print performances to csv
                    for metric in dataset.metrics:
                        e.print_performance_to_csv(model_name=model_name,
                                                   dataset_name=dataset.name,
                                                   performance=metric.value,
                                                   metric_name=metric.name)
            
            if len(models) >= MAX_NUM_MODELS_IN_CACHE:
                self._remove_model_from_cache(framework, model_name)

        logger.info("Finished evaluation.")
        return model_names
