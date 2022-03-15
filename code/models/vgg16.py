import torch
from torch import nn

class FCBlock(nn.Module):

    def __init__(self, in_features, out_features):
        super(FCBlock, self).__init__()
        self.b = nn.Sequential(
            nn.Linear(in_features = in_features, out_features = out_features),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.b(x)

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_bn, kernel_size = 3, padding = 1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding)
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.use_bn = use_bn

    def forward(self, x):
        y = self.conv(x)
        if self.use_bn:
            y = self.bn(y)
        y = self.relu(y)
        return y

class VGG16(nn.Module):

    def __init__(self, n_classes, use_bn, fc_dropout = 0):

        super(VGG16, self).__init__()

        self.name = "vgg16"

        self.n_classes = n_classes
        self.fc_dropout = fc_dropout
        self.drop = self.fc_dropout > 0

        self.b11 = ConvBlock(in_channels = 3, out_channels = 64, use_bn = use_bn)
        self.b12 = ConvBlock(in_channels = 64, out_channels = 64, use_bn = use_bn)
        self.mp1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.b21 = ConvBlock(in_channels = 64, out_channels = 128, use_bn = use_bn)
        self.b22 = ConvBlock(in_channels = 128, out_channels = 128, use_bn = use_bn)
        self.mp2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.b31 = ConvBlock(in_channels = 128, out_channels = 256, use_bn = use_bn)
        self.b32 = ConvBlock(in_channels = 256, out_channels = 256, use_bn = use_bn)
        self.b33 = ConvBlock(in_channels = 256, out_channels = 256, use_bn = use_bn)
        self.mp3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.b41 = ConvBlock(in_channels = 256, out_channels = 512, use_bn = use_bn)
        self.b42 = ConvBlock(in_channels = 512, out_channels = 512, use_bn = use_bn)
        self.b43 = ConvBlock(in_channels = 512, out_channels = 512, use_bn = use_bn)
        self.mp4 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.b51 = ConvBlock(in_channels = 512, out_channels = 512, use_bn = use_bn)
        self.b52 = ConvBlock(in_channels = 512, out_channels = 512, use_bn = use_bn)
        self.b53 = ConvBlock(in_channels = 512, out_channels = 512, use_bn = use_bn)
        self.mp5 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.flatten = nn.Flatten(start_dim = 1, end_dim = -1)

        self.fc1 = FCBlock(512, 128)
        self.drop1 = nn.Dropout(p = fc_dropout)

        self.fc2 = FCBlock(128, 128)
        self.drop2 = nn.Dropout(p = fc_dropout) 

        self.out_layer = nn.Linear(in_features = 128, out_features = n_classes)

    def forward(self, x):

        # x : [b, 3, 32, 32]

        y = self.b11(x) # [b, 64, 32, 32]
        y = self.b12(y) # [b, 64, 32, 32]
        y = self.mp1(y) # [b, 64, 16, 16]

        y = self.b21(y) # [b, 128, 16, 16]
        y = self.b22(y) # [b, 128, 16, 16]
        y = self.mp2(y) # [b, 128, 8, 8]

        y = self.b31(y) # [b, 256, 8, 8]
        y = self.b32(y) # [b, 256, 8, 8]
        y = self.b33(y) # [b, 256, 8, 8]
        y = self.mp3(y) # [b, 256, 4, 4]

        y = self.b41(y) # [b, 512, 4, 4]
        y = self.b42(y) # [b, 512, 4, 4]
        y = self.b43(y) # [b, 512, 4, 4]
        y = self.mp4(y) # [b, 512, 2, 2]

        y = self.b51(y) # [b, 512, 2, 2]
        y = self.b52(y) # [b, 512, 2, 2]
        y = self.b53(y) # [b, 512, 2, 2]
        y = self.mp5(y) # [b, 512, 1, 1]

        y = self.flatten(y) # [b, 512]

        y = self.fc1(y) # [b, 128]
        if self.drop:
            y = self.drop1(y)
        
        y = self.fc2(y) # [b, 128]
        if self.drop:
            y = self.drop2(y)
        
        y = self.out_layer(y) # [b, n_classes]

        return y