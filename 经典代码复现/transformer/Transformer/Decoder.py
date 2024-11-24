import torch
import torch.nn as nn
import Layers


class Decoder_Layer(nn.Module):
    def __init__(self,heads,d_model,dropout=0.1):
        self.maked_attn=Layers.Multihead_Attention_Layer(heads,d_model,dropout)
        self.attn2=Layers.Multihead_Attention_Layer(heads,d_model,dropout)
        self.norm1=Layers.NormLayer(d_model)
        self.norm2=Layers.NormLayer(d_model)
        self.norm3=Layers.NormLayer(d_model)
        self.ffn=Layers.FeedForward(d_model)

    def forward(self,X,e_outputs,src_mask,trg_mask):##src_mask和tar_mask对应两个掩码
        X1=self.maked_attn(X,X,X,trg_mask)# masked attention
        X1=self.norm1(X+X1)
        X2=self.attn2(X1,e_outputs,e_outputs,src_mask)
        X2=self.norm2(X2+X1)
        X3=self.ffn(X2)
        out=self.norm3(X3+X2)
        return out

class Decoder(nn.Module):
    def __init__(self,vocab_size,d_model,N,heads,dropout=0.1):
        super(Decoder,self).__init__()
        self.embed=Layers.Embedding_Layer(d_model,vocab_size)
        self.pe=Layers.Position_Embeding_Layer(d_model)
        self.layers=nn.ModuleList([Decoder_Layer(heads,d_model,dropout) for _ in range(N)])
    def forward(self,X,e_outputs,src_mask,trg_mask):
        X=self.embed(X)
        X=self.pe(X)
        for block in self.layers:
            X=block(X,e_outputs,src_mask,trg_mask)
        return X
