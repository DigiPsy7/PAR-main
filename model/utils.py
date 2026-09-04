from .models import *
from torchvision.models import resnet18, mobilenet_v2
import torch
import torch.nn as nn

def get_model(model_args, training_args, data_args):
    """
    Returns (client_model, server_model) for Split-Federated Learning.
    """
    if model_args.model_name == 'allcnn':
        client_model = ClientAllCNN(n_channels=3)
        server_model = ServerAllCNN(dataset=data_args.dataset_name, num_classes=10)
        
    elif model_args.model_name == 'resnet':
        client_model = ClientResNet()
        server_model = ServerResNet()
        
    elif model_args.model_name == 'mobilenet':
        raise NotImplementedError("Mobilenet split not yet configured - use resnet or allcnn")
    else:
        raise NotImplementedError
        
    return client_model.to(model_args.device), server_model.to(model_args.device)