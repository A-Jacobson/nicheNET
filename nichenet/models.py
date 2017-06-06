import torch.nn as nn


class DadNet(nn.Module):
    def __init__(self):
        super(DadNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=64, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

