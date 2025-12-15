import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dBatchLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, leaka=0.1):
        super(Conv2dBatchLeaky, self).__init__()
        
        # Padding logic for 'same' padding with stride 1 or more
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(negative_slope=leaka)

    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))

class Darknet19(nn.Module):
    def __init__(self):
        super(Darknet19, self).__init__()
        
        self.conv1 = Conv2dBatchLeaky(3, 32, 3, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = Conv2dBatchLeaky(32, 64, 3, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = Conv2dBatchLeaky(64, 128, 3, 1)
        self.conv4 = Conv2dBatchLeaky(128, 64, 1, 1)
        self.conv5 = Conv2dBatchLeaky(64, 128, 3, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv6 = Conv2dBatchLeaky(128, 256, 3, 1)
        self.conv7 = Conv2dBatchLeaky(256, 128, 1, 1)
        self.conv8 = Conv2dBatchLeaky(128, 256, 3, 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv9 = Conv2dBatchLeaky(256, 512, 3, 1)
        self.conv10 = Conv2dBatchLeaky(512, 256, 1, 1)
        self.conv11 = Conv2dBatchLeaky(256, 512, 3, 1)
        self.conv12 = Conv2dBatchLeaky(512, 256, 1, 1)
        self.conv13 = Conv2dBatchLeaky(256, 512, 3, 1)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        self.conv14 = Conv2dBatchLeaky(512, 1024, 3, 1)
        self.conv15 = Conv2dBatchLeaky(1024, 512, 1, 1)
        self.conv16 = Conv2dBatchLeaky(512, 1024, 3, 1)
        self.conv17 = Conv2dBatchLeaky(1024, 512, 1, 1)
        self.conv18 = Conv2dBatchLeaky(512, 1024, 3, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.pool4(x)
        
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        
        # Route 1 (passthrough from here)
        route1 = x
        
        x = self.pool5(x)
        
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        
        return x, route1
