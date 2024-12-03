import torch
import torch.nn as nn

class Wrapper(nn.Module):
    def __init__(self, model, output_layer=0, return_logits=True):
        super(Wrapper, self).__init__() 
        self.model = model
        self.output_layer = output_layer
        self.return_logits = return_logits
    
    def forward(self, x):
        _, list_outputs, logits = self.model(x)
        if self.return_logits:
            return logits if self.output_layer is None else logits[self.output_layer]
        return list_outputs if self.output_layer is None else list_outputs[self.output_layer]
    
resnet50_mlp_simclrnh_local =  dict(    
    local_path="/n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/configural-shape-private/configural_shape_private/logs/runs07_simclr_hn_better_arch/in1k/convnext_tiny_mlp/simclr_hn/20241202_151345"
)

resnet50_mlp_simclrnh_cdac593124 = dict(
    url='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/runs07_simclr_hn_better_arch/in1k/resnet50_mlp/simclr_hn/20241203_055421/final_weights-cdac593124.pth',
    transforms=None, # Add your transforms here
    meta={
        "repo": "https://github.com/harvard-visionlab/alexnets",
        "urls": dict(
            params='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/runs07_simclr_hn_better_arch/in1k/resnet50_mlp/simclr_hn/20241203_055421/params-cdac593124.json',
            train='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/runs07_simclr_hn_better_arch/in1k/resnet50_mlp/simclr_hn/20241203_055421/log_train-cdac593124.txt',
            val='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/runs07_simclr_hn_better_arch/in1k/resnet50_mlp/simclr_hn/20241203_055421/log_val-cdac593124.txt',
        ),
        "_metrics": {},
        "_docs": """
            ....
        """,
    },
)

model_weight_params = dict(
    resnet50_mlp_simclrnh_cdac593124_probe0=resnet50_mlp_simclrnh_cdac593124,
    resnet50_mlp_simclrnh_cdac593124_probe1=resnet50_mlp_simclrnh_cdac593124,
    resnet50_mlp_simclrnh_cdac593124_probe2=resnet50_mlp_simclrnh_cdac593124,
    resnet50_mlp_simclrnh_cdac593124_probe3=resnet50_mlp_simclrnh_cdac593124,

)

def load_model(weights_name, pretrained=True):
    from configural_shape_private.models.weights import Weights, get_standard_transforms
    from configural_shape_private.models import load_model_from_weights, load_model_from_path
    
    weight_dict = model_weight_params[weights_name]
    probe_index = int(weights_name[-1])
    
    if 'local_path' in weight_dict:
        model = Wrapper(load_model_from_path(weight_dict['local_path']), output_layer=probe_index, return_logits=True)
    else:
        if weight_dict['transforms'] is None:
            weight_dict['transforms'] = get_standard_transforms()

        weights = Weights(**weight_dict)
        model = Wrapper(load_model_from_weights(weights), output_layer=probe_index, return_logits=True)
    
    return model