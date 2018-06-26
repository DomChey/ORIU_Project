'''
Implementation of the GoogLeNet as presented in their paper from 2014
https://arxiv.org/abs/1409.4842
We will use this version of the GoogLeNet for our classification task
@author: Dominique Cheray
'''

# Necessary imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# A major building block of the GoogLeNet are the Inception Modules so implement the Module first
class InceptionModule(nn.Module):
    """Implementation of the Inception Module of GoogLeNet 2014"""

    def __init__(self, in_planes, n_1x1, n_red_3x3, n_3x3, n_red_5x5, n_5x5, pool_proj):
        """Initialize the Module with the sizes of the filters used for the convolutions
        and maxpooling

        Args:
            in_planes: number of input filters
            n_1x1: number of filters for the 1x1 conv branch
            n_red_3x3: number of filters in the reduction layer before the 3x3 conv branch
            n_3x3 : number of filters for the 3x3 conv branch
            n_red_5x5: number of filters in the reduction layer before the 5x6 conv branch
            n_5x5: number of filters for the 5x5 conv branch
            pool_proj: number of filters for the 1x1 conv after the built in max-pooling
        """
        super(InceptionModule, self).__init__()

        # 1x1 conv branch
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_planes, n_1x1, kernel_size=1),
            nn.BatchNorm2d(n_1x1),
            nn.ReLU(True)
        )

        # 1x1 conv, 3x3 conv branch
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_planes, n_red_3x3, kernel_size=1),
            nn.BatchNorm2d(n_red_3x3),
            nn.ReLU(True),
            nn.Conv2d(n_red_3x3, n_3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_3x3),
            nn.ReLU(True)
        )

        # 1x1 conv, 5x5 conv branch
        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_planes, n_red_5x5, kernel_size=1),
            nn.BatchNorm2d(n_red_5x5),
            nn.ReLu(True),
            nn.Conv2d(n_red_5x5, n_5x5, kernel_size=5, padding=1),
            nn.BatchNorm2d(n_5x5),
            nn.ReLU(True)
        )

        # 3x3 maxpool, 1x1 conv branch
        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(True)
        )

        def forward(self, x):
            """Forward pass through the Inception Module

            Pass the input through the different branches and then
            concatenate the output of the branches
            """
            out_1 = self.branch_1(x)
            out_2 = self.branch_2(x)
            out_3 = self.branch_3(x)
            out_4 = self.branch_4(x)
            return torch.cat([out_1, out_2, out_3, out_4], 1)


# now that the Inception Module is implemented go ahead and use it for the
# implementation of the GoogLeNet
class GoogLeNet(nn.Module):
    """"Implementation of the GoogLeNet as presented 2014"""
    def __init__(self, n_classes):
        """Build the network according to the 2014 paper
        https://arxiv.org/abs/1409.4842
        """
        super(GoogLeNet, self).__init__()

        # the first layers are not yet inception modules
        self.first_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True)
        )

        # the Inception Modules are occasionally followed by maxpooling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # now build the different Inception Modules
        self.inception_3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)

        self.inception_4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)

        self.inception_5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        # like the first layers the last layers aren't Interception modules
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout2d(p=0.4, inplace=True)
        self.last_linear = nn.Linear(1024, n_classes)

    def forward(self, x):
        """Forward pass through GooǵLeNet"""
        out = self.first_layers(x)
        out = self.maxpool(out)
        out = self.inception_3a(out)
        out = self.inception_3b(out)
        out = self.maxpool(out)
        out = self.inception_4a(out)
        out = self.inception_4b(out)
        out = self.inception_4c(out)
        out = self.inception_4d(out)
        out = self.inception_4e(out)
        out = self.maxpool(out)
        out = self.inception_5a(out)
        out = self.inception_5b(out)
        out = self.avgpool(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.last_linear(out)
        return F.softmax(out, dim=-1)

    def get_features(self, x):
        """Additional helper function to easier get the features of the
        samples for the SVM task of the project"""
        out = self.first_layers(x)
        out = self.maxpool(out)
        out = self.inception_3a(out)
        out = self.inception_3b(out)
        out = self.maxpool(out)
        out = self.inception_4a(out)
        out = self.inception_4b(out)
        out = self.inception_4c(out)
        out = self.inception_4d(out)
        out = self.inception_4e(out)
        out = self.maxpool(out)
        out = self.inception_5a(out)
        out = self.inception_5b(out)
        out = self.avgpool(out)
        return out
