# -*- coding: utf-8 -*-
"""
Autor: Matheus Becali
Email: matheusbecali@gmail.com
"""

import torch
from torch import nn
from metablock import MetaBlock
from metanet import MetaNet
import warnings

# Timm
import timm


class MyRegnety32 (nn.Module):

    def __init__(self, model, num_class, neurons_reducer_block=256, freeze_conv=False, p_dropout=0.5,
                 comb_method=None, comb_config=None, n_feat_conv=1512):

        super(MyRegnety32, self).__init__()

        _n_meta_data = 0
        self.comb = None

        # self.features = nn.Sequential(*list(model.children())[:-1])
        self.model = model

        # .head.fc
        # freezing the convolution layers
        if freeze_conv:
            for param in self.features.parameters():
                param.requires_grad = False

        # Feature reducer
        if comb_method == 'concat':
            warnings.warn("You're using concat with neurons_reducer_block=0. Make sure you're doing it right!")
        self.reducer_block = None

        # Here comes the extra information (if applicable)
        # self.classifier = nn.Sequential(nn.Linear(n_feat_conv + _n_meta_data, num_class))
        # print(model.head.fc)
        self.model.head.fc = nn.Sequential(nn.Flatten(),
                                      nn.Linear(n_feat_conv + _n_meta_data, num_class))


    def forward(self, img, meta_data=None):

        # Checking if when passing the metadata, the combination method is set
        if meta_data is not None and self.comb is None:
            raise Exception("There is no combination method defined but you passed the metadata to the model!")
        if meta_data is None and self.comb is not None:
            raise Exception("You must pass meta_data since you're using a combination method!")

        # x = self.features(img)

        # if self.comb == None:
        #     x = x.view(x.size(0), -1) # flatting
        #     print(x.size())
        #     if self.reducer_block is not None:
        #         x = self.reducer_block(x)  # feat reducer block

        return self.model(img)
