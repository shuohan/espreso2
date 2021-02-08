#!/usr/bin/env python

import torch
from torch import nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros([1, 16, 5, 1]).float())
        print(self.weight.shape)
        self.conv = nn.Conv2d(16, 1, (3, 1))

    def get_kernel(self):
        return self.conv(self.weight)
    def forward(self, x):
        kernel = self.get_kernel()
        return F.conv2d(x, kernel)


net = Network().cuda()
optim = torch.optim.SGD(net.parameters(), lr=0.1)

x = torch.randn(1, 1, 9, 9).float().cuda()
y = net(x)

optim.zero_grad()
# loss = torch.sum(y)
loss = torch.sum(net.get_kernel())
loss.backward()
optim.step()

print(net.weight)

torch.save(net.state_dict(), 'checkpoint.pt')

net.load_state_dict(torch.load('checkpoint.pt'))

x = torch.randn(1, 1, 9, 9).float().cuda()
y = net(x)

optim.zero_grad()
# loss = torch.sum(y)
kernel = net.get_kernel()
loss = torch.sum(kernel)
loss.backward()
optim.step()

print(net.weight)
