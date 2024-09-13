import torch.nn as nn
from pdb import set_trace

class Wrapper(nn.Module):
    def __init__(self, model, probes=None, output_layer=2):
        super(Wrapper, self).__init__()
        self.model = model
        self.probes = probes
        self.output_layer = output_layer
        
    def forward(self, x):
        feat, layer_outputs = self.model.forward_features(x)
        return feat[0]
    
model_weight_params = dict(    
    vit_tiny_patch16_224_mlp_supscram = dict(
        url='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/train_ssl_runs05_vits_supervised/in1k/vit_tiny_patch16_224_mlp/supervised/20240909_131151/final_weights-be616bc1b8.pth',
        transforms=None, # Add your transforms here
        meta={
            "repo": "https://github.com/harvard-visionlab/alexnets",
            "urls": dict(
                params='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/train_ssl_runs05_vits_supervised/in1k/vit_tiny_patch16_224_mlp/supervised/20240909_131151/params-be616bc1b8.json',
                train='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/train_ssl_runs05_vits_supervised/in1k/vit_tiny_patch16_224_mlp/supervised/20240909_131151/log_train-be616bc1b8.txt',
                val='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/train_ssl_runs05_vits_supervised/in1k/vit_tiny_patch16_224_mlp/supervised/20240909_131151/log_val-be616bc1b8.txt',
            ),
            "_metrics": {},
            "_docs": """
                ....
            """,
        },
    )
)

def load_model(weights_name, pretrained=True):
    from configural_shape_private.models.weights import Weights, get_standard_transforms
    from configural_shape_private.models import task_networks, load_model_from_weights
        
    weights = Weights(**model_weight_params[weights_name])
    
    params = weights.get_params_from_url(weights.meta['urls']['params'])
    arch = params['model.arch']
    loss = params['training.loss']
    fc_num_classes = 1000 # params['data.num_classes']
    loss_kwargs = {k.replace("loss_kwargs.",""):v for k,v in params.items() if k.startswith("loss_kwargs") and v is not None}
    model = task_networks.__dict__[arch](loss=loss, fc_num_classes=fc_num_classes, loss_kwargs=loss_kwargs)
    
    if pretrained:
        state_dict = weights.get_state_dict()
        # trim the extra "scrambled image" classes
        state_dict['fc.weight'] = state_dict['fc.weight'][0:1000,:] 
        state_dict['fc.bias'] = state_dict['fc.bias'][0:1000]
        msg = model.load_state_dict(state_dict)
        print(f"{weights_name}: {msg}")
    
    # model = load_model_from_weights(weights, probe_layer=0, return_probe_logits_only=True)
    model = Wrapper(model)
    
    return model

if __name__ == "__main__":
    model = load_model("vit_tiny_patch16_224_mlp_supscram")
