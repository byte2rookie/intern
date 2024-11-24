import torch
import torch.nn as nn
import Layers

class Decoder_Layer(nn.Module):
    def __init__(self,d_model,heads,dropout=0.1):
        super(Decoder_Layer,self).__init__()
        self.d_model=d_model
        self.heads=heads
        self.attn1=Layers.Multihead_Attention(d_model,heads)
        self.attn2=Layers.Multihead_Attention(d_model,heads)
        self.norm1=Layers.Layer_Norm()
        self.norm2=Layers.Layer_Norm()
        self.norm3=Layers.Layer_Norm()
        self.ffn=Layers.FeedForward()

    def forward(self,X,e_output,mask1,mask2):
        """
        :param X:Decoder输入
        :param e_output: Encoder输入
        :param mask1: 第一层attention的掩码
        :param mask2: 第二层attention的编码
        :return: output
        """
        out1=self.attn1(X,X,X,mask1)
        out1=out1+X
        out2=self.norm1(out1)
        out2=self.attn2(e_output,e_output,out2,mask2)
        out2=out2+out1
        out3=self.norm2(out2)
        out4=self.ffn(out3)
        result=out4+out3
        return  result

class Decoder(nn.Module):
    def __init__(self,N,vocab_size,d_model,heads,dropout=0.1):
        """
        :param N: Decoder层数
        :param vocab_size: output输入编码词维度
        :param d_model: 编码后词向量维度
        :param heads: 多头注意力头数
        :param dropout:
        """
        self.N=N
        self.vocab_size=vocab_size
        self.d_model=d_model
        self.heads=heads
        self.embed=Layers.Embeding(vocab_size,d_model)
        self.pe=Layers.Position_Embeding(d_model)
        self.Layers=nn.ModuleList([Decoder_Layer(d_model,heads,dropout) for _ in range(N)])


    def forward(self,X,e_output,mask1,mask2):
        inputs=self.embed(X)
        out=self.pe(inputs)
        for layer in self.Layers:
            out=layer(X,e_output,mask1,mask2)
        return out