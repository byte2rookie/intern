import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class Embedding_Layer(nn.Module):
    def __init__(self,d_model,vocab):
        """
        :param d_model:词向量的维度
        :param vocab: 词表大小(embedding使用时arg1就是用来规定的，输入的词不能超过vocab)
        """
        super(Embedding_Layer,self).__init__()
        self.d_model=d_model
        self.embed=nn.Embedding(vocab,d_model) ##实现Embedding操作,这里定义的embeding矩阵
    def forward(self,X):
        """
        :param X:
        输入的是一个向量
        它可以是原始输入的[['哈','哈','嘻']，['核','核','和']]这样的语句映射到词表形成的向量
        假设vocab=["哈"=0，“嘻”=1,"核"=2，"和"=3]
        X就是[[0,0,1],[2,2,3]]这样的向量形式
        当然也可以用one-hot编码
        这样就是[[[1,0,0,0],[1,0,0,0],[0,1,0,0]],[[0,0,1,0],[0,0,1,0],[0,0,0,1]]]
        这样的输入，也就是(batch,seq_len,code)
        batch不用多说,seq_len表示每个batch输入的语句数，code就是对应的编码了
        而我们这里采用的就是输入one-hot编码
        """
        return self.embed(X)

class Position_Embeding_Layer(nn.Module):
    def __init__(self,d_model,maxlen=5000):
        super(Position_Embeding_Layer,self).__init__()
        position_embed=torch.zeros(maxlen,d_model)  # 实现的是传入向量维度d_model 和最大序列长度max_len
        position=torch.arange(0,maxlen).unsqueeze(1).float()#升高一维，对应tensor多出来的一维
        div_term=torch.exp(torch.arange(0,d_model,2).float()*-(torch.log(torch.tensor(10000.0))/d_model))
        # #根据公式 pe奇数=cos(pos/10000**(2i/d_model))
        # for pos in range(maxlen):
        #     for i in range(d_model):
        #         if i%2 == 1:
        #             position_embed[pos, i]=torch.cos(pos/(10000**(2*i/d_model)))
        #         else:
        #             position_embed[pos, i]=torch.sin(pos/(10000**(2*i/d_model)))

        ##更新后的奇数偶数位置
        position_embed[:,0::2]=torch.sin(position*div_term)
        position_embed[:,1::2]=torch.cos(position*div_term)

        position_embed = position_embed.unsqueeze(0) # [1, maxlen, d_model],增加了batch维度，更好的衔接之前batch输入的word_embedding

        # pe.requires_grad = False
        self.register_buffer('pe', position_embed)
    def forward(self,X):
        seq_len=X.size(1)
        return X+self.pe[:,:seq_len,:]  ##返回一个X+pe计算后[0:X.size[0]]这段的嵌入编码，也就是最终的input

class Multihead_Attention_Layer(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


#LayerNorm
class NormLayer(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        # 层归一化包含两个可以学习的参数
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff=2048,dropout=0.1):
        super(FeedForward,self).__init__()
        self.ffn=nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
            nn.Linear(d_ff,d_model)
        )
        self.dropout=nn.Dropout(dropout)

    def forward(self,X):
        X=self.ffn(X)
        X=self.dropout(X)
        return X

class Masked_Multihead_Attention(nn.Module):
    def __init__(self):
        super(Multihead_Attention_Layer,self).__init__()
    def forward(X):
        return Multihead_Attention_Layer(X)

class OutLayer(nn.Module):
    def __init__(self):
        super(OutLayer,self).__init__()
    def forward(X):
        return OutLayer(X)

