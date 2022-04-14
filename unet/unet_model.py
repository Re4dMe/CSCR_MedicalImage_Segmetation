""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torchvision.models as models
from .unet_parts import *
import numpy as np
from .resunet import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        with torch.no_grad():
            self.res_model = UNet_Resnet50()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        
        #model_ft = models.resnet50(pretrained=True)
        #self.r = nn.Sequential(*(list(model_ft.children())[:-6]))
        #self.inc = DoubleConv(n_channels, 64)
        self.inc = DoubleConv(2, 16)
        self.inc2 = DoubleConv(16, 32)
        self.inc3 = DoubleConv(32, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64 * factor, bilinear)
        self.up5 = Up(64, 64 * factor, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.res_model(x)
        
        #x1 = self.r(x)
        x1_1 = self.inc(x1)
        
        x1_2 = self.inc2(x1_1)
        
        x1_3 = self.inc3(x1_2)
        
        #x2 = self.down1(x1_3)
        #x1 = self.inc(x1_3)
        x2 = self.down1(x1_3)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1_3)
        logits = self.outc(x)
        return logits
        
        
"""
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64 * factor, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

"""