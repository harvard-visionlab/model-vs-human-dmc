import torch
import torch.nn as nn

model_names = dict(
    alexnet_anime_sgd_lr005="hybrid_anime_alexnet_sgd_lr005"
)

def load_model(weights_name, pretrained=True):
    model_name = model_names[weights_name]
    train_type = "catsup"
    model, image_transforms = torch.hub.load('robsymowen/Project-Line', model_name,
                                             force_reload=False, trust_repo=True)
    
    return model