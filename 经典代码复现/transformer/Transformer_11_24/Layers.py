import math

import torch
import torch.nn as nn
import torch.nn.functional as F
class Embeding(nn.Module):
    def __init__(self,vocab_size,d_model):
        """

        :param vocab_size:输入的input对应的词表大小，比如onehot对应的编码就要有vocab_size的维度
        :param d_model: 编码后的输出维度

        Input形式举例
        比如输入了两句话[["哈"，“欸”，“呃”],[“一”,“而”,“说”]]
        映射到词表onehot编码就是[[[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]],[[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]]
        对应的shape就是(1,2,3,6) 2是batch_size 3是seq_len，表示一句话有3个词 6是词向量的映射维度
        """
        super(Embeding,self).__init__()
        self.d_model=d_model
        self.embed=nn.Embedding(vocab_size,d_model)
    def forward(self,X):
        return self.embed(X)

class Position_Embeding(nn.Module):
    def __init__(self,d_model,maxlen=5000):
        """
        :param d_model:embeding维度，不必多说
        :param maxlen: 对应的是一次位置编码最多有多少个词向量
        我们采用绝对位置编码，在编码后将这部分编码内容加入到之前embeding层的部分
        用来表示位置关系对每个词向量的影响
        按照公式
        pe(2i+1) = cos(pos/(10000**(2i/d_model)))
        pe(2i)=sin(pos/10000**(2i/d_model))
        """
        super(Position_Embeding,self).__init__()
        pos_embeding=torch.zeros(maxlen,d_model) # 创建一个manlen*d_model的向量，来表示我们的位置编码
        position=torch.arange(0,maxlen).unsqueeze() #所有的postion
        div_term=torch.exp(torch.arange(0,maxlen,2).float()*-torch.log(torch.tensor(10000))/d_model) #分母项相同

        pos_embeding[:,0::2]=torch.sin(position*div_term)
        pos_embeding[:,1::2]=torch.cos(position*div_term)
        pos_embeding=pos_embeding.unsqueeze(0) #pos变性为(batch_size,seqlen,d_model)的形式
        self.register_buffer('pe',pos_embeding) #将pe编码放入前向传播中

    def forward(self,X):
        seqlen=X.size(1)
        X=X+self.pe[:,:seqlen,:] #对最大长度进行编码，避免额外编码
        return X

class Multihead_Attention(nn.Module):
    def __init__(self,d_model,heads,dropout=0.1):
        """
        :param d_model:传入原本词向量的维度
        :param heads: 分成多少heads
        :param mask: 如果有masked 就masked处理
        :param dropout: 正则化
        """
        super(Multihead_Attention,self).__init__()
        self.d_model=d_model
        self.heads=heads
        self.d_k=d_model//heads

        self.Q_linear=nn.Linear(d_model,d_model) #大Q矩阵
        self.K_linear=nn.Linear(d_model,d_model)
        self.V_linear=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)
        self.out=nn.Linear(d_model,d_model) #这个是将所有的z拼接起来以后再进行一个linear操作输出，更加通用处理

    def attention(self,q,k,v,d_k,mask=None,dropout=None):
        """
        采用attention function来模块化包装计算attention的过程
        :param q: 传入的q矩阵
        :param k: 传入的k矩阵
        :param v: 传入的v矩阵
        :param d_k: 对应每个头的大小维度d_k ，同时用作scale用
        :param mask: 是否mask操作
        :param dropout: 正则化
        :return: 处理好的Z矩阵
        """
        scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(d_k) #计算scale处理后的attention_score
        if mask is not None:
            mask=mask.unsqueeze(1) #mask传入的格式为(batch_size,seqlen) 扩张为(batch_size,1,seqlen)可以更好下一步操作
            scores=torch.masked_fill(mask==0,1e-9)
        scores=F.softmax(scores,dim=-1) #对得到的scores进行softmax操作
        if dropout is not None:
            scores=dropout(scores)
        out=torch.matmul(scores,v)
        return out

    def forward(self,q,k,v,mask=None):
        """
        前向传播计算过程
        :param q: 传入的大q
        :param k: 传入的大k
        :param v: 传入的大v
        :param mask: 掩码
        :return: Z
        """
        batch_size=q.size(0)
        q=self.Q_linear(q).float().view(batch_size,-1,self.heads,self.d_k).transpose(1,2) #且分多头的过程
        k=self.K_linear(k).float().view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)  # 且分多头的过程
        v=self.V_linear(v).float().view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)  # 且分多头的过程

        #接下来对所有的heads计算attention
        z_=self.attention(q,k,v,self.d_k,mask,self.dropout)
        #把所有得到的z拼接到一起
        Z=z_.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)
        out=self.out(Z)
        return out

class Layer_Norm(nn.Module):
    def __init__(self):
        self.norm=nn.LayerNorm()
    def forward(self,X):
        return self.norm(X)

class FeedForward(nn.Module):
    def __init__(self,d_model,dff=2048,dropout=0.1):
        self.linear1=nn.Linear(d_model,dff)
        self.relu=nn.ReLU()
        self.linear2=nn.Linear(dff,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,X):
        out=self.linear1(X)
        out=self.relu(out)
        out=self.dropout(out)
        out=self.linear2(out)
        return out


