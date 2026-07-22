import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.init as init
import numpy as np

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3):
        super(UNetConvBlock, self).__init__()

        self.conv = nn.Conv3d(in_size, out_size, kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv3d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        init.xavier_uniform_(self.conv.weight, gain = np.sqrt(2.0))
        init.constant_(self.conv.bias,0)
        init.xavier_uniform_(self.conv2.weight, gain = np.sqrt(2.0))
        init.constant_(self.conv2.bias,0)
    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))

        return out

class ResUNetConvBlock_2D(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3):
        super(ResUNetConvBlock_2D, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.in_size = in_size
        self.out_size = out_size
        if in_size != out_size:
            self.convX = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0)
            init.xavier_uniform_(self.convX.weight, gain=np.sqrt(2.0))
            init.constant_(self.convX.bias, 0)

        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        init.xavier_uniform_(self.conv.weight, gain=np.sqrt(2.0))
        init.constant_(self.conv.bias, 0)
        init.xavier_uniform_(self.conv2.weight, gain=np.sqrt(2.0))
        init.constant_(self.conv2.bias, 0)


    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        if self.in_size!=self.out_size:
            bridge = self.activation(self.convX(x))
        else:
            bridge = x
        output = torch.add(out, bridge)

        return output

class ResUNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3):
        super(ResUNetConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_size, out_size, kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv3d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.in_size = in_size
        self.out_size = out_size
        if in_size != out_size:
            self.convX = nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0)
            init.xavier_uniform_(self.convX.weight, gain=np.sqrt(2.0))
            init.constant_(self.convX.bias, 0)

        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        init.xavier_uniform_(self.conv.weight, gain=np.sqrt(2.0))
        init.constant_(self.conv.bias, 0)
        init.xavier_uniform_(self.conv2.weight, gain=np.sqrt(2.0))
        init.constant_(self.conv2.bias, 0)


    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        if self.in_size!=self.out_size:
            bridge = self.activation(self.convX(x))
        else:
            bridge = x
        output = torch.add(out, bridge)

        return output

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_size, in_size, kernel_size=(4,3,3), stride=(2,1,1), padding=(1,1,1))
        self.up1 = nn.Conv3d(in_size, out_size, kernel_size, stride=1, padding=1)
        self.conv = nn.Conv3d(in_size, out_size, kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv3d(out_size, out_size, kernel_size, stride=1, padding=1)

        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        init.xavier_uniform_(self.up1.weight, gain = np.sqrt(2.0))
        init.constant_(self.up1.bias,0)
        init.xavier_uniform_(self.conv.weight, gain = np.sqrt(2.0))
        init.constant_(self.conv.bias,0)
        init.xavier_uniform_(self.conv2.weight, gain = np.sqrt(2.0))
        init.constant_(self.conv2.bias,0)


    def forward(self, x, bridge):
        up = self.up1(self.up(x))
        up = self.activation(up)
        out = torch.cat([up, bridge], 1)
        out = self.activation(self.conv(out))
        out = self.activation(self.conv2(out))

        return out

class ResUNetUpBlock_2D(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3):
        super(ResUNetUpBlock_2D, self).__init__()

        self.up = nn.Conv2d(in_size, out_size, kernel_size, stride=1, padding=1)
        init.xavier_uniform_(self.up.weight, gain = np.sqrt(2.0))
        init.constant_(self.up.bias,0)
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.resUnit = ResUNetConvBlock_2D(in_size, out_size, kernel_size=kernel_size)

    def forward(self, x, bridge):
        up = self.activation(self.up(x))
        out = torch.cat([up, bridge], 1)
        out = self.resUnit(out)

        return out


class ResUNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3):
        super(ResUNetUpBlock, self).__init__()

        self.up = nn.ConvTranspose3d(in_size, in_size, kernel_size=(4,3,3), stride=(2,1,1), padding=(1,1,1))
        self.up1 = nn.Conv3d(in_size, out_size, kernel_size, stride=1, padding=1)
        init.xavier_uniform_(self.up1.weight, gain = np.sqrt(2.0))
        init.constant_(self.up1.bias,0)
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.resUnit = ResUNetConvBlock(in_size, out_size, kernel_size=kernel_size)

    def forward(self, x, bridge):
        up = self.activation(self.up1(self.up(x)))
        out = torch.cat([up, bridge], 1)
        out = self.resUnit(out)

        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EncoderBlock, self).__init__()
        self.downsample = nn.Conv3d(in_channel, in_channel, kernel_size=(3,3,3), stride=(2,1,1), padding=(1,1,1))
        self.conv = ResUNetConvBlock(in_channel, out_channel)

    def forward(self, x):
        down = self.downsample(x)
        out = self.conv(down)
        return down, out

class ResUnet2D(nn.Module):
    def __init__(self, in_channel=1, f_maps=[16, 32, 64]):
        super(ResUnet2D, self).__init__()
        self.in_channel = in_channel
        self.f_maps = f_maps
        self.f_maps_de = f_maps[::-1]
        self.num_layers = len(self.f_maps)
        self.encoder = nn.ModuleList([])
        self.initconv = ResUNetConvBlock_2D(self.in_channel,f_maps[0])
        for i in range(self.num_layers-1):
            self.encoder.append(ResUNetConvBlock_2D(self.f_maps[i],self.f_maps[i+1]))
        self.bottle_neck = ResUNetConvBlock_2D(self.f_maps[-1],self.f_maps[-1])
        self.decoder = nn.ModuleList([])
        for i in range(self.num_layers-1):
            self.decoder.append(ResUNetUpBlock_2D(self.f_maps_de[i],self.f_maps_de[i+1]))
        self.final_conv = nn.Conv2d(self.f_maps_de[-1], in_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        encoders_features = []
        x = self.initconv(x)
        encoders_features.insert(0,x)
        for encoder in self.encoder:
            x = encoder(x)
            encoders_features.insert(0, x)

        for decoder, encoder_features in zip(self.decoder, encoders_features[1:]):
            x = decoder(x, encoder_features)
        x = self.final_conv(x)

        return x


class ResUnet3D(nn.Module):
    def __init__(self, in_channel=1, f_maps=[16, 32, 64]):
        super(ResUnet3D, self).__init__()
        self.in_channel = in_channel
        self.f_maps = f_maps
        self.f_maps_de = f_maps[::-1]
        self.num_layers = len(self.f_maps)
        self.encoder = nn.ModuleList([])
        self.initconv = ResUNetConvBlock(self.in_channel,f_maps[0])
        for i in range(self.num_layers-1):
            self.encoder.append(EncoderBlock(self.f_maps[i],self.f_maps[i+1]))
        self.bottle_neck = ResUNetConvBlock(self.f_maps[-1],self.f_maps[-1])
        self.decoder = nn.ModuleList([])
        for i in range(self.num_layers-1):
            self.decoder.append(ResUNetUpBlock(self.f_maps_de[i],self.f_maps_de[i+1]))
        self.final_conv = nn.Conv3d(self.f_maps_de[-1], in_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        encoders_features = []
        x = self.initconv(x)
        encoders_features.insert(0,x)
        for encoder in self.encoder:
            downsample, x = encoder(x)
            encoders_features.insert(0, x)

        for decoder, encoder_features in zip(self.decoder, encoders_features[1:]):
            x = decoder(x, encoder_features)
        x = self.final_conv(x)

        return x





if __name__ == '__main__':
    a = torch.zeros([1,1,16,128,128]).transpose(0,2).squeeze(1)
    model = ResUnet2D(f_maps=[16,32,64])
    b = model(a)
    print(b.shape)