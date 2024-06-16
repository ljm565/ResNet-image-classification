import torch.nn as nn
import torch.nn.functional as F



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