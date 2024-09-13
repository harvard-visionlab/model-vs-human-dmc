'''
    python run.py plotting <plotting_def_name>
    
    e.g.,
    python run.py plotting plotting_definition_run04_vits
    python run.py plotting plotting_definition_run05_vits
    
    Figures are output to repo/figures/plotting_def_name
'''
import fire
import warnings
import plotting_definitions
warnings.simplefilter(action='ignore', category=FutureWarning)

from modelvshuman import Plot, Evaluate
from modelvshuman import constants as c
from pdb import set_trace

import argparse
parser = argparse.ArgumentParser(description='Model vs Human scripts')

FLAGS, FIRE_FLAGS = parser.parse_known_args()

def testing():
    print("Hello world.")

def plotting(plotting_def_name, 
             plot_types=c.DEFAULT_PLOT_TYPES # or e.g., ["accuracy", "shape-bias"]
):
    plotting_def = plotting_definitions.__dict__[plotting_def_name]
    figure_dirname = f"./{plotting_def_name}/"
    Plot(plot_types = plot_types, plotting_definition = plotting_def,
         figure_directory_name = figure_dirname)

    # In examples/plotting_definition.py, you can edit
    # plotting_definition_template as desired: this will let
    # the toolbox know which models to plot, and which colours to use etc.


if __name__ == "__main__":
    fire.Fire(command=FIRE_FLAGS)