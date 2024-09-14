'''
    python run.py plotting <plotting_def_name>
    
    e.g.,
    python run.py evaluation
    
    python run.py plotting plotting_definition_run04_vits
    python run.py plotting plotting_definition_run05_vits
    
    Figures are output to repo/figures/plotting_def_name
'''
import os
import fire
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import plotting_definitions
from modelvshuman_dmc import Plot, Evaluate
from modelvshuman_dmc import constants as c
from pdb import set_trace

import argparse
parser = argparse.ArgumentParser(description='Model vs Human scripts')

FLAGS, FIRE_FLAGS = parser.parse_known_args()

def evaluation(batch_size=64, print_predictions=True, num_workers=len(os.sched_getaffinity(0))):
    # models = ["alexnet", "resnet50", "bagnet33", "simclr_resnet50x1", "vit_b_16", "convnext_large"]
    models = ["alexnet"]
    datasets = ["colour"] # c.DEFAULT_DATASETS # or e.g. ["cue-conflict", "uniform-noise"]
    params = {"batch_size": batch_size, "print_predictions": print_predictions, "num_workers": num_workers}
    Evaluate()(models, datasets, **params)

def plotting(plotting_def_name="plotting_definition_template", 
             plot_types=c.DEFAULT_PLOT_TYPES # or e.g., ["accuracy", "shape-bias"]
):
    plotting_def = plotting_definitions.__dict__[plotting_def_name]
    figure_dirname = f"./{plotting_def_name}/"
    Plot(plot_types = plot_types, plotting_definition = plotting_def,
         figure_directory_name = figure_dirname)

    # In examples/plotting_definitions/plotting_definition.py, you can edit
    # plotting_definition_template as desired: this will let
    # the toolbox know which models to plot, and which colours to use etc.


if __name__ == "__main__":
    fire.Fire(command=FIRE_FLAGS)