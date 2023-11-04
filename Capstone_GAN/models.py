import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.5):
        super(ResNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels))

        # Shortcut connection (if necessary)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        # Main path
        out = self.block(x)
        # Shortcut path
        shortcut = self.shortcut(x)
        # Add the main path and the shortcut path
        out += shortcut
        return out
    

class Encoder(nn.Module):
    def __init__(self, input_shape):
        self.model = nn.Sequential(
            #Downsampling
            nn.Conv2d(6, 64, 7),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #9 resnet blocks
            ResNetBlock(in_channels=256, out_channels=256),
            ResNetBlock(in_channels=256, out_channels=256),
            ResNetBlock(in_channels=256, out_channels=256),
            ResNetBlock(in_channels=256, out_channels=256),
            ResNetBlock(in_channels=256, out_channels=256),
            ResNetBlock(in_channels=256, out_channels=256),
            ResNetBlock(in_channels=256, out_channels=256),
            ResNetBlock(in_channels=256, out_channels=256),
            ResNetBlock(in_channels=256, out_channels=256),
            #Upsampling
            nn.ConvTranspose2d(256, 128, 3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 7),
            nn.Tanh())

    def forward(self, x):
        out = self.model(x)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid())

    def forward(self, x):
        out = self.model(x)
        return out

class Decoder(nn.Module):
    def __init__(self):
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.Conv2d(64, 126, 3, padding=1),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid())

    def forward(self, x):
        out = self.model(x)
        return out
