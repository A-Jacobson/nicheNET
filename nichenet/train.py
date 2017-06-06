import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable

from nichenet.datasets import RasterData
from nichenet import config
from nichenet.models import DadNet

rasterdata = RasterData(root=config.ROOT)
rasterdata_train = data.DataLoader(rasterdata, num_workers=8,
                                   batch_size=8)

dadnet = DadNet()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(dadnet.parameters())


num_epochs = 200
for epoch in range(num_epochs):
    for i, (rasters, targets) in enumerate(rasterdata_train):
        if i > 0:
            break
        rasters = Variable(rasters)
        targets = Variable(targets)
        optimizer.zero_grad()
        outputs = dadnet(rasters)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print('Epoch [%d/%d], Loss: %.4f'
              % (epoch + 1, num_epochs, loss.data[0]))

