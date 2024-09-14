import torch.nn as nn

class Wrapper(nn.Module):
    def __init__(self, model, probes, output_layer=2):
        super(Wrapper, self).__init__()
        self.model = model
        self.probes = probes
        self.output_layer = output_layer
        
    def forward(self, x):
        feat, layer_outputs = self.model.forward_features(x)
        output = self.probes[self.output_layer](layer_outputs[self.output_layer])
        return output
    
model_weight_params = dict(
    vit_tiny_patch16_224_mlp_configclr2_alpha090 =  dict(url='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/train_ssl_runs04_vits/in1k/vit_tiny_patch16_224_mlp/configclr2/20240904_202704/final_weights-2b7b899437.pth',
        transforms=None, # Add your transforms here
        meta={
            "repo": "https://github.com/harvard-visionlab/alexnets",
            "urls": dict(
                params='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/train_ssl_runs04_vits/in1k/vit_tiny_patch16_224_mlp/configclr2/20240904_202704/params-2b7b899437.json',
                train='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/train_ssl_runs04_vits/in1k/vit_tiny_patch16_224_mlp/configclr2/20240904_202704/log_train-2b7b899437.txt',
                val='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/train_ssl_runs04_vits/in1k/vit_tiny_patch16_224_mlp/configclr2/20240904_202704/log_val-2b7b899437.txt',
            ),
            "_metrics": {},
            "_docs": """
                ....
            """,
        },
    ),
    vit_base_patch16_224_mlp_simclr  = dict(url='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/train_ssl_runs04_vits/in1k/vit_base_patch16_224_mlp/simclr/20240905_101137/final_weights-71a8cd399b.pth',
        transforms=None, # Add your transforms here
        meta={
            "repo": "https://github.com/harvard-visionlab/alexnets",
            "urls": dict(
                params='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/train_ssl_runs04_vits/in1k/vit_base_patch16_224_mlp/simclr/20240905_101137/params-71a8cd399b.json',
                train='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/train_ssl_runs04_vits/in1k/vit_base_patch16_224_mlp/simclr/20240905_101137/log_train-71a8cd399b.txt',
                val='https://s3.wasabisys.com/visionlab-members/alvarez/Projects/configural_shape_private/train_ssl_runs04_vits/in1k/vit_base_patch16_224_mlp/simclr/20240905_101137/log_val-71a8cd399b.txt',
            ),
            "_metrics": {},
            "_docs": """
                ....
            """,
        },
    )    
)

def load_model(model_name, pretrained=True):
    from configural_shape_private.models.weights import Weights, get_standard_transforms
    from configural_shape_private.models import task_networks, load_model_from_weights
    
    weights_name, output_layer = "_".join(model_name.split("_")[0:-1]), int(model_name.split("_")[-1])
    
    weights = Weights(**model_weight_params[weights_name])
    
    params = weights.get_params_from_url(weights.meta['urls']['params'])
    arch = params['model.arch']
    loss = params['training.loss']
    fc_num_classes = params['data.num_classes']
    loss_kwargs = {k.replace("loss_kwargs.",""):v for k,v in params.items() if k.startswith("loss_kwargs") and v is not None}
    model = task_networks.__dict__[arch](loss=loss, fc_num_classes=fc_num_classes, loss_kwargs=loss_kwargs)
    
    if pretrained:
        state_dict = weights.get_state_dict()
        msg = model.load_state_dict(state_dict)
        print(f"{model_name}: {msg}")
    
    model = load_model_from_weights(weights, probe_layer=0, return_probe_logits_only=True)
    model = Wrapper(model.model, model.probes, output_layer)
    
    return model