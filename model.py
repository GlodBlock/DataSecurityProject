import torch
from torch import nn


class Inc(nn.Module):

    def __init__(self):
        super(Inc, self).__init__()
        # 16x16x192 -> 16x16x64
        self.br1 = nn.Conv2d(192, 64, 1, 1)
        # 16x16x192 -> 16x16x128
        self.br2 = nn.Sequential(
            # 16x16x192 -> 16x16x96
            nn.Conv2d(192, 96, 1, 1),
            nn.ReLU(),
            # 16x16x96 -> 16x16x128
            nn.Conv2d(96, 128, 3, 1, 1)
        )
        # 16x16x192 -> 16x16x32
        self.br3 = nn.Sequential(
            # 16x16x192 -> 16x16x16
            nn.Conv2d(192, 16, 1, 1),
            nn.ReLU(),
            # 16x16x16 -> 16x16x32
            nn.Conv2d(16, 32, 5, 1, 2)
        )
        # 16x16x192 -> 16x16x32
        self.br4 = nn.Sequential(
            # 16x16x192 -> 16x16x192
            nn.MaxPool2d((3, 3), 1, 1),
            # 16x16x192 -> 16x16x32
            nn.Conv2d(192, 32, 1, 1)
        )

    def forward(self, x):
        br1 = self.br1(x)
        br2 = self.br2(x)
        br3 = self.br3(x)
        br4 = self.br4(x)
        return torch.cat([br1, br2, br3, br4], 1)


class VPN_CNN(nn.Module):

    def __init__(self):
        super(VPN_CNN, self).__init__()
        self.layer1 = nn.Sequential(
            # 32x32x1 -> 32x32x64
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            # 32x32x64 -> 16x16x64
            nn.MaxPool2d((2, 2), 2)
        )
        self.layer2 = nn.Sequential(
            # 16x16x64 -> 16x16x192
            nn.Conv2d(64, 192, 1, 1),
            nn.ReLU()
        )
        # 16x16x192 -> 16x16x256
        self.inc = Inc()
        self.layer3 = nn.Sequential(
            # 16x16x256 -> 14x14x512
            nn.Conv2d(256, 512, 3, 1),
            nn.ReLU(),
            # 14x14x512 -> 7x7x512
            nn.MaxPool2d((2, 2), 2)
        )
        self.layer4 = nn.Sequential(
            # 7x7x512 -> 7x7x1024
            nn.Conv2d(512, 1024, 1, 1),
            nn.ReLU(),
            # 7x7x1024 -> 1x1x1024
            nn.AvgPool2d((7, 7))
        )
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, 2)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.inc(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        if y is not None:
            y = y.long()
            loss = self.loss(x, y)
            return x, loss
        else:
            return x
