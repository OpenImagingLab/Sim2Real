import torch
from torch import load, save
from os.path import join
from tools.registery import MODEL_REGISTRY
import torch.nn as nn


def deparse_weights(weights_path):
    state_dict = load(weights_path)
    weights = state_dict['model_state']
    epoch = state_dict['epoch']
    metrics = state_dict['metrics']
    return weights, epoch, metrics


def save_model(epoch, metrics, params, model):
    if epoch % params.validation_config.weights_save_freq == 0:
        model_state = model.state_dict()
        save_dict = {
            'model_state':model_state,
            'epoch':epoch,
            'metrics':metrics,
        }
        save(save_dict, join(params.paths.save.weights, f"{params.model_config.name}_{epoch}.pt"))

def weights_partial_loading(model, weights):
    model_state = model.state_dict()
    for k in list(model_state.keys()):
        # if "evs_correction" in k: continue
        model_state.update({
            k:weights[k]
        })
    return model_state

# def weights_partial_loading(model, weights):
#     model_state = model.state_dict()
#     for k in list(model_state.keys()):
#         if "evs_correction" in k:
#             continue
#         if k in weights:
#             model_state.update({k: weights[k]})
#             param = model.get_parameter(k)
#             if param is not None:
#                 param.requires_grad = False
#     return model_state

def deparse_model(params):
    model = MODEL_REGISTRY.get(params.model_config.name)(params).cuda()

    # for m in model.modules():
    #     if isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #     elif isinstance(m, nn.BatchNorm2d):
    #         nn.init.constant_(m.weight, 1)
    #         nn.init.constant_(m.bias, 0)

    if 'gpu_num' in params.keys():
        print("*** Multiple OPU")
        model.net = torch.nn.DataParallel(model.net)
       
    if params.model_config.model_pretrained is not None:
        weights, epoch, metrics = deparse_weights(params.model_config.model_pretrained)
        if params.enable_training:
            # epoch += 1
            epoch = 0
        model.load_state_dict(weights_partial_loading(model, weights))
        # model.load_state_dict(weights)
        print(f'Weights successfully load from {params.model_config.model_pretrained}\nAfter loading, the epoch is: {epoch}')
    else:
        epoch = 0
        metrics = {}
    for k in params.training_config.losses.keys():
        if f"train_{k}" not in metrics:
            metrics.update({
                f"train_{k}":[]
            })
    for k in params.validation_config.losses.keys():
        if f"val_{k}" not in metrics:
            metrics.update({
                f"val_{k}":[]
            })
    model._init_metrics(metrics)

    return model, epoch, metrics