import torch
import torch.nn as nn
import Layers
import Encoder
import Decoder

class Transformer(nn.Module):
    def __init__(self,N1,N2,vocab_size,d_model,heads,dropout):
        super(Transformer,self).__init__()
        self.Encoder_N=N1
        self.Decoder_N=N2
        self.vocab_size=vocab_size
        self.d_model=d_model
        self.heads=heads
        self.Encoder=Encoder.Encoder(self.Encoder_N,vocab_size,d_model,heads,dropout)
        self.Decoder=Decoder.Decoder(self.Decoder_N,vocab_size,d_model,heads,dropout)
        self.out_linear=nn.Linear(d_model,vocab_size)
        self.softmax=nn.Softmax()
    def forward(self,X,en_mask,out_mask1,out_mask2):
        encoder_out=self.Encoder(X,en_mask)
        decoder_out=self.Decoder(X,encoder_out,out_mask1,out_mask2)
        out=self.out_linear(decoder_out)
        out=self.softmax(out)
        return out
