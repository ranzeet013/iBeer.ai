import torch
from torchvision import models
import torch.nn as nn
from collections import OrderedDict


class BeerModelInference(nn.Module):
    def __init__(self):
        super(BeerModelInference, self).__init__()
        self.network = models.vgg16(pretrained=True)
        self.network.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, 512)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.5)),
            ('fc2', nn.Linear(512, 10)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

    def forward(self, xb):
        return self.network(xb)


def load_model(model_path, device):
    model = BeerModelInference().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
