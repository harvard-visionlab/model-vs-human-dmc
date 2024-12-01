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
    
alexnet_w1_mlp_simclrhn_7df7cb689c =  dict(    
        url='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/runs06_simclr_hn/in1k/alexnet_w1_mlp/simclr_hn/20241129_093123/final_weights-7df7cb689c.pth',
        transforms=None, # Add your transforms here
        meta={
            "repo": "https://github.com/harvard-visionlab/alexnets",
            "urls": dict(
                params='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/runs06_simclr_hn/in1k/alexnet_w1_mlp/simclr_hn/20241129_093123/params-7df7cb689c.json',
                train='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/runs06_simclr_hn/in1k/alexnet_w1_mlp/simclr_hn/20241129_093123/log_train-7df7cb689c.txt',
                val='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/runs06_simclr_hn/in1k/alexnet_w1_mlp/simclr_hn/20241129_093123/log_val-7df7cb689c.txt',
            ),
            "_metrics": {},
            "_docs": """
                ....
            """,
        },
    )

alexnet_w3_mlp_simclrhn_822826223d =  dict(    
        url='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/runs06_simclr_hn/in1k/alexnet_w3_mlp/simclr_hn/20241129_122633/final_weights-822826223d.pth',
        transforms=None, # Add your transforms here
        meta={
            "repo": "https://github.com/harvard-visionlab/alexnets",
            "urls": dict(
                params='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/runs06_simclr_hn/in1k/alexnet_w3_mlp/simclr_hn/20241129_122633/params-822826223d.json',
                train='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/runs06_simclr_hn/in1k/alexnet_w3_mlp/simclr_hn/20241129_122633/log_train-822826223d.txt',
                val='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/runs06_simclr_hn/in1k/alexnet_w3_mlp/simclr_hn/20241129_122633/log_val-822826223d.txt',
            ),
            "_metrics": {},
            "_docs": """
                ....
            """,
        },
    
alexnet_lgn2_w1_mlp_simclrhn_62fe9c9bf5 = dict(
        url='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/runs06_simclr_hn/in1k/alexnet_lgn2_w1_mlp/simclr_hn/20241201_074149/final_weights-62fe9c9bf5.pth',
        transforms=None(), # Add your transforms here
        meta={
            "repo": "https://github.com/harvard-visionlab/alexnets",
            "urls": dict(
                params='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/runs06_simclr_hn/in1k/alexnet_lgn2_w1_mlp/simclr_hn/20241201_074149/params-62fe9c9bf5.json',
                train='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/runs06_simclr_hn/in1k/alexnet_lgn2_w1_mlp/simclr_hn/20241201_074149/log_train-62fe9c9bf5.txt',
                val='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/runs06_simclr_hn/in1k/alexnet_lgn2_w1_mlp/simclr_hn/20241201_074149/log_val-62fe9c9bf5.txt',
            ),
            "_metrics": {},
            "_docs": """
                ....
            """,
        },
    )

)

alexnet_lgn6_w1_mlp_simclrhn_2fc9823322 = dict(
    url='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/runs06_simclr_hn/in1k/alexnet_lgn6_w1_mlp/simclr_hn/20241201_074739/final_weights-2fc9823322.pth',
    transforms=None, # Add your transforms here
    meta={
        "repo": "https://github.com/harvard-visionlab/alexnets",
        "urls": dict(
            params='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/runs06_simclr_hn/in1k/alexnet_lgn6_w1_mlp/simclr_hn/20241201_074739/params-2fc9823322.json',
            train='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/runs06_simclr_hn/in1k/alexnet_lgn6_w1_mlp/simclr_hn/20241201_074739/log_train-2fc9823322.txt',
            val='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/runs06_simclr_hn/in1k/alexnet_lgn6_w1_mlp/simclr_hn/20241201_074739/log_val-2fc9823322.txt',
        ),
        "_metrics": {},
        "_docs": """
            ....
        """,
    },
)

model_weight_params = dict(
    alexnet_w1_mlp_simclrhn_probe0=alexnet_w1_mlp_simclrhn_7df7cb689c,
    alexnet_w1_mlp_simclrhn_probe1=alexnet_w1_mlp_simclrhn_7df7cb689c,
    alexnet_w1_mlp_simclrhn_probe2=alexnet_w1_mlp_simclrhn_7df7cb689c,
    alexnet_w1_mlp_simclrhn_probe3=alexnet_w1_mlp_simclrhn_7df7cb689c,
    alexnet_w3_mlp_simclrhn_probe0=alexnet_w3_mlp_simclrhn_822826223d,
    alexnet_w3_mlp_simclrhn_probe1=alexnet_w3_mlp_simclrhn_822826223d,
    alexnet_w3_mlp_simclrhn_probe2=alexnet_w3_mlp_simclrhn_822826223d,
    alexnet_w3_mlp_simclrhn_probe3=alexnet_w3_mlp_simclrhn_822826223d,
    
    alexnet_lgn2_w1_mlp_simclrhn_probe0=alexnet_lgn2_w1_mlp_simclrhn_62fe9c9bf5,
    alexnet_lgn2_w1_mlp_simclrhn_probe1=alexnet_lgn2_w1_mlp_simclrhn_62fe9c9bf5,
    alexnet_lgn2_w1_mlp_simclrhn_probe2=alexnet_lgn2_w1_mlp_simclrhn_62fe9c9bf5,
    alexnet_lgn2_w1_mlp_simclrhn_probe3=alexnet_lgn2_w1_mlp_simclrhn_62fe9c9bf5,
    
    alexnet_lgn6_w1_mlp_simclrhn_probe0=alexnet_lgn6_w1_mlp_simclrhn_2fc9823322,
    alexnet_lgn6_w1_mlp_simclrhn_probe1=alexnet_lgn6_w1_mlp_simclrhn_2fc9823322,
    alexnet_lgn6_w1_mlp_simclrhn_probe2=alexnet_lgn6_w1_mlp_simclrhn_2fc9823322,
    alexnet_lgn6_w1_mlp_simclrhn_probe3=alexnet_lgn6_w1_mlp_simclrhn_2fc9823322,
)

def load_model(weights_name, pretrained=True):
    from configural_shape_private.models.weights import Weights, get_standard_transforms
    from configural_shape_private.models import load_model_from_weights
    
    weight_dict = model_weight_params[weights_name]
    if weight_dict['transforms'] is None:
        weight_dict['transforms'] = get_standard_transforms()
        
    weights = Weights(**weight_dict)
    
    probe_index = int(weights_name[-1])
    model = Wrapper(load_model_from_weights(weights), output_layer=probe_index, return_logits=True)
    
    return model