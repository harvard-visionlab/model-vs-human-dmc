import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from modelvshuman import Plot, Evaluate
from modelvshuman import constants as c
from plotting_definition import plotting_definition_template


def run_evaluation():
    models = [
        # "vit_tiny_patch16_224_mlp_configclr2_alpha090_0", 
        # "vit_tiny_patch16_224_mlp_configclr2_alpha090_1", 
        # "vit_tiny_patch16_224_mlp_configclr2_alpha090_2", 
        # "vit_tiny_patch16_224_mlp_configclr2_alpha090_3",
        # "vit_base_patch16_224_mlp_simclr_0",
        # "vit_base_patch16_224_mlp_simclr_1",
        # "vit_base_patch16_224_mlp_simclr_2",
        # "vit_base_patch16_224_mlp_simclr_3",
        "vit_tiny_patch16_224_mlp_supscram",
        # "vit_b_16",
    ]
    datasets = c.DEFAULT_DATASETS # or e.g. ["cue-conflict", "uniform-noise"]
    params = {"batch_size": 64, "print_predictions": True, "num_workers": 20}
    Evaluate()(models, datasets, **params)

def run_plotting():
    plot_types = c.DEFAULT_PLOT_TYPES # or e.g. ["accuracy", "shape-bias"]
    plotting_def = plotting_definition_template
    figure_dirname = "./examples/example-figures/evaluate_vits_runs04/"
    Plot(plot_types = plot_types, plotting_definition = plotting_def,
         figure_directory_name = figure_dirname)

    # In examples/plotting_definition.py, you can edit
    # plotting_definition_template as desired: this will let
    # the toolbox know which models to plot, and which colours to use etc.


if __name__ == "__main__":
    # 1. evaluate models on out-of-distribution datasets
    run_evaluation()
    # 2. plot the evaluation results
    # run_plotting()
