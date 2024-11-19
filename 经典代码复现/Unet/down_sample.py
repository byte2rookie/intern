import torch.nn as nn
import torch


# 根据论文图像可知，有DownSample,Upsample,pooling,concat,1x1Conv 五种操作
# 首先定义DownSample操作



class DownSample_Layer(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(DownSample_Layer,self).__init__()
        self.Conv_BN_ReLU=nn.Sequential(
            ##尊重原文方式
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        self.DownSample=nn.Sequential(
            nn.MaxPool2d(kernel_size=2)
        )
    def forward(self,x):
        out=self.Conv_BN_ReLU(x)    #用于后续的concat
        result=self.DownSample(out)
        return out,result
