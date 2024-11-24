
import torch
import torch.nn as nn
import Layers
import Encoder
import Decoder
class Transformer(nn.Module):
    def __init__(self,vocab_size,trg_size,d_model,N,heads,dropout):
        super(Transformer,self).__init__()
        self.encoder=Encoder(vocab_size,d_model,N,heads,dropout)
        self.decoder=Decoder(vocab_size,d_model,N,heads,dropout)
        self.linear=nn.Linear(d_model,trg_size)
        self.out=nn.Softmax()
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs=self.encoder(src,src_mask)
        d_output=self.decoder(trg,e_outputs,src_mask,trg_mask)
        output=self.linear(d_output)
        output=self.out(output)
        return output

