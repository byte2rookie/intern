import torch
import torch.nn as nn


def crop_feature_map(encoder_feature, decoder_feature):
    """
    裁剪 encoder_feature，使其大小匹配 decoder_feature。

    Args:
        encoder_feature (torch.Tensor): 高分辨率特征图，形状如 (N, C, H_enc, W_enc)。
        decoder_feature (torch.Tensor): 目标低分辨率特征图，形状如 (N, C, H_dec, W_dec)。

    Returns:
        torch.Tensor: 裁剪后的高分辨率特征图，形状匹配 decoder_feature。
    """
    _, _, h_dec, w_dec = decoder_feature.size()
    _, _, h_enc, w_enc = encoder_feature.size()

    # 计算裁剪的起始索引
    crop_h_start = (h_enc - h_dec) // 2
    crop_w_start = (w_enc - w_dec) // 2

    # 裁剪特征图
    cropped = encoder_feature[:, :, crop_h_start:crop_h_start + h_dec, crop_w_start:crop_w_start + w_dec]
    return cropped


class UpSample_Layer(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(UpSample_Layer,self).__init__()
        self.Conv_BN_ReLU=nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel*2,kernel_size=3),
            nn.BatchNorm2d(out_channel*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*2, kernel_size=3),
            nn.BatchNorm2d(out_channel*2),
            nn.ReLU(),
        )
        self.UpSampler=nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_channel*2,out_channels=out_channel,kernel_size=2,stride=2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )


    def forward(self,X,out):
        #X是上一层传递的初始输入，out则是对应的下采样层的卷积输出（对应DownSample的out）
        out1=self.Conv_BN_ReLU(X)
        out2=self.UpSampler(out1)
        con_out=crop_feature_map(encoder_feature=out,decoder_feature=out2)
        out3=torch.concat((out2,con_out),dim=1)
        return out3
