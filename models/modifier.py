import torch


def replace_batchnorm2d_by_instancenorm2d(module: torch.nn.Module):
    """Given a module, replaces batch normalization by instance normalization"""
    for name, layer in module.named_children():
        if isinstance(layer, torch.nn.BatchNorm2d):
            norm_layer = torch.nn.InstanceNorm2d(num_features=layer.num_features,
                                                 eps=layer.eps,
                                                 momentum=layer.momentum,
                                                 affine=layer.affine,
                                                 track_running_stats=layer.track_running_stats)
            norm_layer.load_state_dict(layer.state_dict())
            setattr(module, name, norm_layer)
        replace_batchnorm2d_by_instancenorm2d(layer)
