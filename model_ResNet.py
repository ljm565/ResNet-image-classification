import torch.nn as nn
import torch.nn.functional as F
from config import Config


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, down_sample, zero_padding):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.down_sample = down_sample
        self.zero_padding = zero_padding
        if self.down_sample and not self.zero_padding:
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=self.stride, padding=0, bias=False),
                nn.BatchNorm2d(self.out_channels)
            )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels)
        )
        self.relu = nn.ReLU()

    
    def downSampling(self, x):
        """
        pad the 3 dimensional tensor except batch
        padding = (left, rigt, top, bottom, front, back)
        """
        if self.zero_padding:   
            padding = (0, 0, 0, 0, 0, self.out_channels - self.in_channels)
            out = F.pad(x, padding)
            out = nn.MaxPool2d(kernel_size=2, stride=2)(out)
            return out
        return self.conv1x1(x)


    def forward(self, x):
        shortcut = self.downSampling(x) if self.down_sample else x

        # first conv layer
        out = self.conv1(x)
        out = self.relu(out)

        # second conv layer and residual connection
        out = self.conv2(out)          
        out = self.relu(out + shortcut)

        return out



class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, down_sample, zero_padding):
        super(CNNBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels)
        )
        self.relu = nn.ReLU()


    def forward(self, x):
        # first conv layer
        out = self.conv1(x)
        out = self.relu(out)

        # second conv layer
        out = self.conv2(out)          
        out = self.relu(out)

        return out



class ResNet(nn.Module):
    def __init__(self, config:Config, color_channel:int, num_layer:int, block):
        super(ResNet, self).__init__()
        self.height = config.height
        self.width = config.width
        assert self.height == self.width
        self.label = config.label
        self.color_channel = color_channel
        self.num_layer = num_layer
        self.block = block
        self.zero_padding = config.zero_padding

        # first conv layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.color_channel, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # residual layers
        self.conv2x = self.get_layers(in_channels=16, out_channels=16, stride=1, block=self.block)
        self.conv3x = self.get_layers(in_channels=16, out_channels=32, stride=2, block=self.block)
        self.conv4x = self.get_layers(in_channels=32, out_channels=64, stride=2, block=self.block)

        # last conv layer
        self.avg_pool = nn.AvgPool2d(kernel_size=int(self.height/4), stride=1, padding=0)
        self.fc = nn.Linear(int(self.height/4)**2, self.label)

        # initialization
        self.init_wts()
        

    def init_wts(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def get_layers(self, in_channels, out_channels, stride, block):
        down_sample = False if stride == 1 else True
        layer_list = [block(in_channels, out_channels, stride, down_sample, self.zero_padding)]
        for _ in range(self.num_layer-1):
            layer_list.append(block(out_channels, out_channels, 1, False, self.zero_padding))
        layer_list = nn.ModuleList(layer_list)
        return nn.Sequential(*layer_list)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2x(x)
        x = self.conv3x(x)
        x = self.conv4x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x