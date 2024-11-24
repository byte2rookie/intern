import torch
import torch.nn as nn
import Layers



class Encoder_Layer(nn.Module):
    def __init__(self,d_model,heads,dropout):
        super(Encoder_Layer,self).__init__()
        self.attention=Layers.Multihead_Attention_Layer(heads,d_model,dropout)
        self.ffn=Layers.FeedForward(d_model)
        self.norm1=Layers.NormLayer(d_model)
        self.norm2=Layers.NormLayer(d_model)

    def forward(self,X):
        X1=self.attention(X,X,X)
        X1=self.norm1(X+X1)
        X2=self.ffn(X1)
        out=self.norm2(X2+X1)
        return out

class Encoder(nn.Module):
    def __init__(self,vocab_size,d_model,N,heads,dropout):
        super(Encoder,self).__init__()
        self.N=N
        self.embed=Layers.Embedding_Layer(d_model,vocab_size)
        self.pe=Layers.Position_Embeding_Layer(d_model)
        self.blocks=nn.ModuleList([Encoder_Layer(d_model,heads,dropout) for _ in range(N)])

    def forward(self,src,mask):
        x=self.embed(src)
        x=self.pe(x)
        for block in self.blocks:
            x=block(x,mask)
        return x

