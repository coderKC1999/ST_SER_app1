import torch
import torch.nn as nn

import timm

from torchvision import models


class HybridResNetViT(nn.Module):

    def __init__(self, num_classes=8):

        super().__init__()

        self.cnn = models.resnet18(weights=None)

        self.cnn.fc = nn.Identity()

        self.vit = timm.create_model(

            "vit_tiny_patch16_224",

            pretrained=False,

            num_classes=0,

            global_pool="avg"

        )

        fused_dim = 512 + self.vit.num_features

        self.classifier = nn.Sequential(

            nn.Linear(fused_dim,512),

            nn.ReLU(),

            nn.Dropout(0.6),

            nn.Linear(512,num_classes)

        )

    def forward(self,x):

        cnn_f = self.cnn(x)

        vit_f = self.vit(x)

        fused = torch.cat(

            [cnn_f,vit_f],

            dim=1

        )

        out = self.classifier(

            fused

        )

        return out
