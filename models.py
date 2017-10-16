import torch.nn as nn
import torch.nn.functional as F


class ConvBatchRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvBatchRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.batchnorm(x), True)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBatchRelu(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = ConvBatchRelu(out_channels, out_channels, kernel_size, padding=padding)
        self.conv1x1 = ConvBatchRelu(in_channels, out_channels, 1)

    def forward(self, x):
        fx = self.conv1(x)
        fx = self.conv2(fx)
        return fx + self.conv1x1(x)




class DadNet(nn.Module):
    """
    89.6% accuracy on bird dataset
    """
    def __init__(self):
        super(DadNet, self).__init__()
        self.batch_norm = nn.BatchNorm2d(7)
        self.conv1 = ConvBatchRelu(7, 14, 3, 1)
        self.conv2 = ConvBatchRelu(14, 14, 3, 1)
        self.conv3 = ConvBatchRelu(14, 28, 3, 1)
        self.conv4 = ConvBatchRelu(28, 28, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(28 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), True)
        x = self.fc2(x)
        return x


class DadNetv2(nn.Module):
    """
    89.4% accuracy on bird dataset 13 epochs
    """
    def __init__(self):
        super(DadNetv2, self).__init__()
        self.batch_norm = nn.BatchNorm2d(7)
        self.pool = nn.AvgPool2d(2)
        self.res1 = ResBlock(7, 14, 3, 1)
        self.res2 = ResBlock(14, 28, 3, 1)
        self.fc1 = nn.Linear(28 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.res1(x)
        x = self.pool(x)
        x = self.res2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), True)
        x = self.fc2(x)
        return x

