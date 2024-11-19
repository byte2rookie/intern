import torch
import torch.nn as nn
import down_sample
import up_sample
class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.d1=down_sample.DownSample_Layer(1,64)
        self.d2=down_sample.DownSample_Layer(64,128)
        self.d3=down_sample.DownSample_Layer(128,256)
        self.d4=down_sample.DownSample_Layer(256,512)

        self.u1=up_sample.UpSample_Layer(512,512)
        self.u2=up_sample.UpSample_Layer(1024,256)
        self.u3 = up_sample.UpSample_Layer(512, 128)
        self.u4 = up_sample.UpSample_Layer(256,64)

        self.out=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=2,kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self,X):
        out_1, out1 = self.d1(X)
        out_2, out2 = self.d2(out1)
        out_3, out3 = self.d3(out2)
        out_4, out4 = self.d4(out3)
        out5 = self.u1(out4, out_4)
        out6 = self.u2(out5, out_3)
        out7 = self.u3(out6, out_2)
        out8 = self.u4(out7, out_1)
        out = self.out(out8)
        return out

if __name__ == '__main__':
    X_shape=(1,1,572,572)
    X=torch.rand(X_shape)
    unet1=Unet()
    out=unet1(X)
    print(out.shape)