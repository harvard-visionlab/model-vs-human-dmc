import torch.nn as nn

class Wrapper(nn.Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        output = self.model(x)
        return output[0]
    
model_weight_params = dict(
    alexnet2023_baseline_pgd =  dict(    
        url='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/model_rearing_workshop/models/in1k/alexnet2023_baseline/supervised/20231202_060532/final_weights-5bb4b657e4.pth',
        transforms=None, # Add your transforms here
        meta={
            "repo": "https://github.com/harvard-visionlab/model-rearing-workshop",
            "urls": dict(
                params='https://s3.us-east-1.wasabisys.com/visionlab-members/alvarez/Projects/model_rearing_workshop/models/in1k/alexnet2023_baseline/supervised/20231202_060532/params-5bb4b657e4.json',
                train='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/model_rearing_workshop/models/in1k/alexnet2023_baseline/supervised/20231202_060532/log_train-5bb4b657e4.txt',
                val='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/model_rearing_workshop/models/in1k/alexnet2023_baseline/supervised/20231202_060532/log_val-5bb4b657e4.txt',
            ),
            "_metrics": {},
            "_docs": """
                ....
            """,
        },
    ),
)

def load_model(weights_name, pretrained=True):
    from model_rearing_workshop.models import load_model_from_weights
    from model_rearing_workshop.models.weights import Weights, get_standard_transforms
    
    weights = Weights(**model_weight_params[weights_name])
    model = Wrapper(load_model_from_weights(weights, return_probed_layer_activations=False, return_probe_logits=False))
    
    return model