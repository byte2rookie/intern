import torch
import torch.nn as nn
import Layers
class Encoder_Layer(nn.Module):
    def __init__(self,d_model,heads,mask,dropout):
        super(Encoder_Layer,self).__init__()
        self.mask=mask
        self.d_model=d_model
        self.heads=heads
        self.attention = Layers.Multihead_Attention(d_model,heads,mask,dropout)
        self.ffn=Layers.FeedForward(d_model,dropout)
        self.norm1=Layers.Layer_Norm()
        self.norm2=Layers.Layer_Norm()
    def forward(self,X):
        out1=self.attention(X,X,X,self.mask)
        out1=out1+X
        out1=self.norm1(out1)
        out2=self.ffn(out1)
        out2=out2+out1
        out2=self.norm2(out2)
        return out2

class Encoder(nn.Module):
    def __init__(self,N,vocab_size,d_model,heads,dropout):
        """

        :param N:层数为N
        :param vocab_size:原词表维度大小
        :param d_model: 转换后词向量维度大小
        :param heads: 多头头数
        :param mask: 掩码
        :param dropout:
        """
        super(Encoder,self).__init__()
        self.N=N
        self.d_model=d_model
        self.heads=heads,
        self.embed=Layers.Embeding(vocab_size,d_model)
        self.pe=Layers.Position_Embeding(d_model)
        self.Layers=nn.ModuleList([Encoder_Layer(d_model,heads,dropout) for _ in range(N)])

    def forward(self,X,mask):
        inputs=self.embed(X)
        output=self.pe(inputs)
        for layer in self.Layers:
            output=layer(output,mask)
        return output
