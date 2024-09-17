#!/usr/bin/env python

import os
from os.path import join as pjoin

##################################################################
# DIRECTORIES
##################################################################

PROJ_DIR = str(os.environ.get("MODELVSHUMAN_DMC_DIR", "model-vs-human-dmc"))
assert (PROJ_DIR != "None"), "Please set the 'MODELVSHUMAN_DMC_DIR' environment variable as described in the README"
CODE_DIR = pjoin(PROJ_DIR, "modelvshuman_dmc")
DATASET_DIR = pjoin(PROJ_DIR, "datasets")
OUTPUT_DIR = pjoin(PROJ_DIR, "outputs")
FIGURE_DIR = pjoin(PROJ_DIR, "outputs", "figures")
RAW_DATA_DIR = pjoin(PROJ_DIR, "outputs", "raw-data")
PERFORMANCES_DIR = pjoin(RAW_DATA_DIR, "outputs", "performances")
RESULTS_DIR = pjoin(PROJ_DIR, "outputs", "results")
REPORT_DIR = pjoin(PROJ_DIR, "outputs", "latex-report/")
ASSETS_DIR = pjoin(PROJ_DIR, "assets/")
ICONS_DIR = pjoin(ASSETS_DIR, "icons/")
DECISION_MAPPING = "ImageNetProbabilitiesTo16ClassesMappingWithSortedProbs"

##################################################################
# CONSTANTS
##################################################################

IMG_SIZE = 224  # size of input images for most models

##################################################################
# ANALYSES
##################################################################

DEFAULT_ANALYSES = [
    "humanvshuman_splithalves_noise_ceiling",
    "humanvshuman_pairwise_accuracy_correlation",
    "humanvshuman_error_consistency",
    "modelvsmodel_pairwise_decision_margin_consistency",
    "modelvsmodel_pairwise_error_consistency",    
]

EXPECTED_SUBJECTS = {
    "edge": 10,
    "silhouette": 10,
    "cue-conflict": 10,
    "sketch": 7,
    "stylized": 5,
}

ANALYSIS_MODEL_GROUPS = dict(
    demo=["alexnet", "resnet50", "bagnet33", "simclr_resnet50x1", "vit_b_16", "convnext_large"]  
)
##################################################################
# DATASETS
##################################################################

NOISE_GENERALISATION_DATASETS = ["colour",
                                 "contrast",
                                 "high-pass",
                                 "low-pass",
                                 "phase-scrambling",
                                 "power-equalisation",
                                 "false-colour",
                                 "rotation",
                                 "eidolonI",
                                 "eidolonII",
                                 "eidolonIII",
                                 "uniform-noise"]

TEXTURE_SHAPE_DATASETS = ["original", "greyscale",
                          "texture", "edge", "silhouette",
                          "cue-conflict"]

DEFAULT_DATASETS = ["edge", "silhouette", "cue-conflict"] + \
                   NOISE_GENERALISATION_DATASETS + ["sketch", "stylized"]
##################################################################
# PLOT TYPES
##################################################################

PLOT_TYPE_TO_DATASET_MAPPING = {
    # default plot types:
    "shape-bias": ["cue-conflict"],
    "accuracy": NOISE_GENERALISATION_DATASETS,
    "nonparametric-benchmark-barplot": ["edge", "silhouette", "sketch", "stylized"],
    "benchmark-barplot": DEFAULT_DATASETS,
    "scatterplot": DEFAULT_DATASETS,
    "error-consistency-lineplot": NOISE_GENERALISATION_DATASETS,
    "error-consistency": ["cue-conflict", "edge", "silhouette", "sketch", "stylized"],
    # 'unusual' plot types:
    "entropy": NOISE_GENERALISATION_DATASETS,
    "confusion-matrix": DEFAULT_DATASETS,
    }

DEFAULT_PLOT_TYPES = list(PLOT_TYPE_TO_DATASET_MAPPING.keys())
DEFAULT_PLOT_TYPES.remove("entropy")
DEFAULT_PLOT_TYPES.remove("confusion-matrix")

##################################################################
# MODELS
##################################################################

TORCHVISION_MODELS = ["alexnet",
                      "vgg11_bn",
                      "vgg13_bn",
                      "vgg16_bn",
                      "vgg19_bn",
                      "squeezenet1_0",
                      "squeezenet1_1",
                      "densenet121",
                      "densenet169",
                      "densenet201",
                      "inception_v3",
                      "resnet18",
                      "resnet34",
                      "resnet50",
                      "resnet101",
                      "resnet152",
                      "shufflenet_v2_x0_5",
                      "mobilenet_v2",
                      "resnext50_32x4d",
                      "resnext101_32x8d",
                      "wide_resnet50_2",
                      "wide_resnet101_2",
                      "mnasnet0_5",
                      "mnasnet1_0"]

BAGNET_MODELS = ["bagnet9", "bagnet17", "bagnet33"]

SHAPENET_MODELS = ["resnet50_trained_on_SIN",
                   "resnet50_trained_on_SIN_and_IN",
                   "resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN"]

SIMCLR_MODELS = ["simclr_resnet50x1", "simclr_resnet50x2", "simclr_resnet50x4"]

PYCONTRAST_MODELS = ["InsDis", "MoCo", "PIRL", "MoCoV2", "InfoMin"]

SELFSUPERVISED_MODELS = SIMCLR_MODELS + PYCONTRAST_MODELS

EFFICIENTNET_MODELS = ["efficientnet_b0", "noisy_student"]

ADV_ROBUST_MODELS = ["resnet50_l2_eps0", "resnet50_l2_eps0_5",
                     "resnet50_l2_eps1", "resnet50_l2_eps3",
                     "resnet50_l2_eps5"]

VISION_TRANSFORMER_MODELS = ["vit_small_patch16_224", "vit_base_patch16_224",
                             "vit_large_patch16_224"]

BIT_M_MODELS = ["BiTM_resnetv2_50x1", "BiTM_resnetv2_50x3", "BiTM_resnetv2_101x1",
                "BiTM_resnetv2_101x3", "BiTM_resnetv2_152x2", "BiTM_resnetv2_152x4"]

SWAG_MODELS = ["swag_regnety_16gf_in1k", "swag_regnety_32gf_in1k", "swag_regnety_128gf_in1k",
               "swag_vit_b16_in1k", "swag_vit_l16_in1k", "swag_vit_h14_in1k"]
