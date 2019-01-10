from torchvision import models
from pretrainedmodels.models import densenet121
from torch import nn
from config import config
from collections import OrderedDict
import torch.nn.functional as F
import types

def my_modify_densenets(model):
    model.features.conv0 = nn.Conv2d(config.channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.Dropout(0.5),
                nn.Linear(1024, config.num_classes),
            )

    def logits(self, features):
        x = model.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def get_net():
    model = densenet121(pretrained="imagenet")
    model = my_modify_densenets(model)
    return model

# model = get_net()
# print(model)