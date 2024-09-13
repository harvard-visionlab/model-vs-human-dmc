# /!usr/bin/env python3

"""
Define decision makers (either human participants or CNN models).
"""

from modelvshuman import constants as c
from modelvshuman.plotting.colors import *
from modelvshuman.plotting.decision_makers import DecisionMaker

__all__ = ['plotting_definition_run05_vits']

def plotting_definition_run05_vits(df):
    """Decision makers to compare a few models with human observers.

    This definition includes specific models, grouped by color
    for easier visualization of categories.
    """

    decision_makers = []

    # Define color shades for each model group
    orange_shades = [rgb(255, 165, 0), rgb(255, 140, 0), rgb(255, 120, 0), rgb(255, 100, 0)]
    blue_shades = [rgb(65, 105, 225), rgb(70, 130, 180), rgb(100, 149, 237), rgb(135, 206, 250)]

    # 1. vit_tiny_patch16_224_mlp_configclr2_alpha090 models (shades of orange)
    # for i, model in enumerate([
    #     "vit_tiny_patch16_224_mlp_configclr2_alpha090_0", 
    #     "vit_tiny_patch16_224_mlp_configclr2_alpha090_1", 
    #     "vit_tiny_patch16_224_mlp_configclr2_alpha090_2", 
    #     "vit_tiny_patch16_224_mlp_configclr2_alpha090_3"
    # ]):
    #     decision_makers.append(DecisionMaker(name_pattern=model,
    #                            color=orange_shades[i], marker="o", df=df,
    #                            plotting_name=f"ViT Tiny ConfigCLR Alpha {i+1}"))

    # 2. vit_base_patch16_224_mlp_simclr models (shades of blue)
    # for i, model in enumerate([
    #     "vit_base_patch16_224_mlp_simclr_0",
    #     "vit_base_patch16_224_mlp_simclr_1",
    #     "vit_base_patch16_224_mlp_simclr_2",
    #     "vit_base_patch16_224_mlp_simclr_3"
    # ]):
    #     decision_makers.append(DecisionMaker(name_pattern=model,
    #                            color=blue_shades[i], marker="o", df=df,
    #                            plotting_name=f"ViT Base SimCLR {i+1}"))

    decision_makers.append(DecisionMaker(name_pattern="vit_base_patch16_224_mlp_supscram",
                           color=rgb(0, 159, 220), marker="o", df=df,
                           plotting_name="ViT-T 16 SupScram"))
    
    # 3. vit_b_16 (reference model)
    decision_makers.append(DecisionMaker(name_pattern="vit_b_16",
                           color=rgb(144, 159, 110), marker="v", df=df,
                           plotting_name="ViT-B 16"))

    # 4. Humans (standard reference)
    decision_makers.append(DecisionMaker(name_pattern="subject-*",
                           color=rgb(165, 30, 55), marker="D", df=df,
                           plotting_name="humans"))

    return decision_makers


def get_comparison_decision_makers(df, include_humans=True, humans_last=True):
    """Decision makers used in our paper with updated colors for new models."""

    d = []
    
    # Define color shades for each model group
    orange_shades = [rgb(255, 165, 0), rgb(255, 140, 0), rgb(255, 120, 0), rgb(255, 100, 0)]
    blue_shades = [rgb(65, 105, 225), rgb(70, 130, 180), rgb(100, 149, 237), rgb(135, 206, 250)]
    
    # 1. vit_tiny_patch16_224_mlp_configclr2_alpha090 models (shades of orange)
    for i, model in enumerate([
        "vit_tiny_patch16_224_mlp_configclr2_alpha090_0", 
        "vit_tiny_patch16_224_mlp_configclr2_alpha090_1", 
        "vit_tiny_patch16_224_mlp_configclr2_alpha090_2", 
        "vit_tiny_patch16_224_mlp_configclr2_alpha090_3"
    ]):
        d.append(DecisionMaker(name_pattern=model,
                               color=orange_shades[i], marker="o", df=df,
                               plotting_name=f"ViT Tiny ConfigCLR Alpha {i+1}"))

    # 2. vit_base_patch16_224_mlp_simclr models (shades of blue)
    for i, model in enumerate([
        "vit_base_patch16_224_mlp_simclr_0",
        "vit_base_patch16_224_mlp_simclr_1",
        "vit_base_patch16_224_mlp_simclr_2",
        "vit_base_patch16_224_mlp_simclr_3"
    ]):
        d.append(DecisionMaker(name_pattern=model,
                               color=blue_shades[i], marker="o", df=df,
                               plotting_name=f"ViT Base SimCLR {i+1}"))

    # 3. vit_b_16 model (reference)
    d.append(DecisionMaker(name_pattern="vit_b_16",
                           color=rgb(144, 159, 110), marker="v", df=df,
                           plotting_name="ViT-B 16"))

    # 4. Humans (standard reference)
    if not humans_last and include_humans:
        d.append(DecisionMaker(name_pattern="subject-*",
                               color=red, marker="D", df=df,
                               plotting_name="humans"))
    
    # 5. Add other models (optional, depending on your use case)

    # 6. Add humans if requested and at the end
    if humans_last:
        if include_humans:
            d.append(DecisionMaker(name_pattern="subject-*",
                                   color=red, marker="D", df=df,
                                   plotting_name="humans"))

    return d
