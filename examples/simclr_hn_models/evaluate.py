import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from modelvshuman_dmc import Plot, Evaluate
from modelvshuman_dmc import constants as c

def run_evaluation():
    models = [
        "alexnet2023_baseline_pgd",
        "alexnet_w1_mlp_simclrhn_probe0",
        "alexnet_w1_mlp_simclrhn_probe1",
        "alexnet_w1_mlp_simclrhn_probe2",
        "alexnet_w1_mlp_simclrhn_probe3",
        "alexnet_w3_mlp_simclrhn_probe0",
        "alexnet_w3_mlp_simclrhn_probe1",
        "alexnet_w3_mlp_simclrhn_probe2",
        "alexnet_w3_mlp_simclrhn_probe3",
    ]
    datasets = c.DEFAULT_DATASETS # or e.g. ["cue-conflict", "uniform-noise"]
    params = {"batch_size": 64, "print_predictions": True, "num_workers": 20}
    Evaluate()(models, datasets, **params)

def run_plotting():
    plot_types = c.DEFAULT_PLOT_TYPES # or e.g. ["accuracy", "shape-bias"]
    plotting_def = plotting_definition_template
    figure_dirname = "./examples/example-figures/evaluate_simclr_hn_models/"
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
