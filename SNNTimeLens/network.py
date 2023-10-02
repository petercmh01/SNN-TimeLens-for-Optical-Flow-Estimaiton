import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from spikingjelly.activation_based import neuron, functional, surrogate, layer

class CSNN(nn.Module):
    def __init__(self, N_in = 6, use_cupy=False):
        super().__init__()

        ### Encoding ###
        self.block1 = nn.Sequential(

        layer.Conv2d(N_in, 32, kernel_size=7, padding=1, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),

        layer.Conv2d(32, 32, kernel_size=7, padding=1, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),)

        # residual out

        layer.MaxPool2d(2, 2),

        self.block2 = nn.Sequential(

        layer.Conv2d(32, 64, kernel_size=5, padding=1, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),

        layer.Conv2d(64, 64, kernel_size=5, padding=1, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),)

        # residual out

        layer.MaxPool2d(2, 2),

        self.block2 = nn.Sequential(

        layer.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),

        layer.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),)

        # residual out

        layer.MaxPool2d(2, 2),

        self.block3 = nn.Sequential(
        layer.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),

        layer.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),)

        # residual out

        layer.MaxPool2d(2, 2),


        self.block4 = nn.Sequential(
        layer.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(512),
        neuron.IFNode(surrogate_function=surrogate.ATan()),

        layer.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),)

        # residual out

        layer.MaxPool2d(2, 2),


        self.block5 = nn.Sequential(
        layer.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),

        layer.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),)

        # residual out

        layer.MaxPool2d(2, 2),

        self.block6 = nn.Sequential(
        layer.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),

        layer.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),)


        ### Decoding ###

        self.block7 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        layer.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),)

        # residual in

        self.block8 = nn.Sequential(
        layer.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),

        nn.Upsample(scale_factor=2, mode='bilinear'),

        layer.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),)

        # resiual in

        self.block9 = nn.Sequential(
        layer.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),

        nn.Upsample(scale_factor=2, mode='bilinear'),

        layer.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),)

        # residual in

        self.block10 = nn.Sequential(
        layer.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),

        nn.Upsample(scale_factor=2, mode='bilinear'),

        layer.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),)

        # residual in

        self.block11 = nn.Sequential(
        layer.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),

        nn.Upsample(scale_factor=2, mode='bilinear'),

        layer.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),)

        ### residual in

        self.block12 = nn.Sequential(
        layer.Conv2d(32, 2, kernel_size=3, padding=1, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),

        layer.Conv2d(2, 2, kernel_size=1, padding=0, bias=False),
        nn.Tanh())


    def forward(self, x):
      e1 = self.block1(x)
      e2 = self.block2(e1)
      e3 = self.block3(e2)
      e4 = self.block4(e3)
      e5 = self.block5(e4)
      e6 = self.block6(e5)
      d7 = self.block7(e6) + e5
      d8 = self.block8(d7) + e4
      d9 = self.block8(d8) + e3
      d10 = self.block8(d9) + e2
      d11 = self.block8(d10) + e1
      d12 = self.block8(d11)

      return d12
