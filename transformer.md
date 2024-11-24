# 目的
搞清楚transformer的原理和实现

## 一、transfomer和其他架构的对比
### CNN
特点：权重共享--同一个卷积核，滑动窗口--滑动计算，相对位置敏感，绝对位置不敏感（比如图片打乱像素排布很大影响，而卷积从后向前和从前向后的顺序则没有影响）
CNN网络等卷积网络可以做到并行计算，其图片等等形式的输入在卷积过程具有平移不变性，也就是从左到右卷积和从右到左卷积得到的结果是相同的

### RNN
对相对位置、绝对位置都很敏感
串行计算
单步计算复杂度不变
O(n) 线性计算复杂度
### transformer
1.相对位置不敏感，可并行计算
2.绝对位置不敏感，如果需要表示绝对位置，就需要对位置进行单独的编码


## 二、self-attention
self-attention作为RNN,CNN之后出现的一种提取信息的方式
其作用是对每一个输入的单元 提取出其与整个句子所有元素之间的相关度
Query,Key,Value 三种input得到的形式
Activation(Query·Key)=Attention Score 也就是相关度
∑（Attention Score * Value）=hidden-output
类比就是CNN 多通道输入 进行CNN卷积核提取信息后，多个channel 卷积提出的信息累加，获得一个feature map
其实self-attention就是卷积思路的拓展
CNN是受限制的self-attention,所以训练时所需的数据量比较小

## 三、解读transfomer与复现
transformer最初用于处理Seq2Seq类型的问题
用Encoder层不断提取信息，得到feature map 传递给Decoder层不断解码，最后输出一串可能的向量
这个跟Unet的思路类似，encoder-decoder
只不过Seq2Seq是不定长输出，而Unet是定长的输出罢了
## 四、transformer架构解读
input=Embeding

Encoder=
(Multihead-Attention + Add&Norm + Feed Forward + Add&Norm)*N

Decoder=
(Masked-Multihead-Attention + Add&Norm + Multihead-Attention+Add&Norm+Feed Forward + Add&Norm)*N

输出outlayer=(fc+softmax)
这样可以把Transformer的框架源码写出来了

我首先写出相关的框架，然后对照别人复现好的模块，一步步去修正我的复现代码，并按照每个操作去学习相关的知识

### 1.Embedding层
参考blog:[关于embedding层的知识](https://blog.csdn.net/m0_37605642/article/details/132866531)
Embedding操作就是将input的words转化为神经网络可以处理的向量形式
常见Embeding操作的有one-hot编码
而这里我们要用的Embedding层，不同于one-hot的稀疏形式，Embedding层将会产生稠密的向量
Embedding向量能够表达对象的某些特征，两个向量之间的距离反映了对象之间的相似性。简单的说，Embedding就是把一个东西映射到一个向量X。如果这个东西很像，那么得到的向量x1和x2的欧式距离很小。
举例说明：
Word Embedding，把单词w映射到向量x。如果两个单词的意思相近，比如bike和bicycle，那么它们映射后得到的两个词向量x1和x2的欧式距离很小。
User Embedding，把用户ID映射到向量x。推荐系统中，需要用一个向量表示一个用户，如果两个用户的行为习惯接近，那么他们对应的向量x1和x2的欧式距离很小。
Graph Embedding，把图中的每个节点映射成一个向量x。如果图中两个节点接近，比如它们的最短路很小，那么它们embed得到的向量x1和x2的欧式距离很小。
多模态里的embedding，就是直接把两个向量通过全连接变成一个。

而transformer的embedding层所做的，就是把稀疏的input 通过Input和W_embedding的矩阵相乘的方式，输出一个更小维度的embedded_input
比如1w*50w的输入 乘一个50w*1的矩阵，就输出了一个1w*1的向量
这个向量作为输入比起原始输入小得多，而embedding矩阵是可以不断学习，从而优化提取向量的质量的，所以叫embedding层

### Position_Embedding层
一句话概括，Positional Encoding就是将位置信息添加（嵌入）到Embedding词向量中，让Transformer保留词向量的位置信息，可以提高模型对序列的理解能力。 
也就是将embedding层的输出向量和Position_Embedding层的输出向量concat起来，获取到最终的输入向量
[参考博客1](https://blog.csdn.net/m0_37605642/article/details/132866365)
[参考博客2](https://blog.csdn.net/python123456_/article/details/141352984)
[参考博客3](https://blog.csdn.net/weixin_41806489/article/details/128403466)
Position_Embedding业界流行有三种实现
1.绝对编码
2.相对编码
3.旋转编码
#### 绝对编码
其实原理就是，输入的有很多单词，我们给每个单词编上号
比如["哈","噶","噶","官"]就对应一个编号[0,1,2,3]
在["哈","噶","噶","官"]输入embedding层产生word_embedding的时候
[0,1,2,3]也根据公式产生一个和词嵌入大小形状相同的嵌入，也就是pos_embedding
pos_embedding在奇数位置和偶数位置的公式不同
```
GPT的示范代码
import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(pos, d_model):
    """
    计算给定位置和模型维度的绝对位置编码。
    """
    encoding = np.zeros(d_model)
    for i in range(d_model):
        if i % 2 == 0:  # 偶数位置用 sin
            encoding[i] = np.sin(pos / (10000 ** (2 * i / d_model)))
        else:  # 奇数位置用 cos
            encoding[i] = np.cos(pos / (10000 ** (2 * (i - 1) / d_model)))
    return encoding

# 参数设定
sentence = "我爱你，中国。"  # 输入句子
d_model = 32  # 模型维度
positions = range(len(sentence))  # 根据句子长度生成位置索引

# 计算所有字符的编码
encodings = np.array([positional_encoding(pos, d_model) for pos in positions])

# 绘图
plt.figure(figsize=(10, 6))
plt.imshow(encodings, aspect='auto', cmap='viridis')
plt.colorbar(label="Encoding Value")
plt.title(f"Absolute Positional Encoding for Sentence: {sentence}", fontsize=14)
plt.xlabel("Model Dimension (d_model)", fontsize=12)
plt.ylabel("Position in Sentence", fontsize=12)
plt.xticks(range(0, d_model, 2))  # 显示每隔两维的维度
plt.yticks(range(len(sentence)), labels=[f"'{char}'" for char in sentence])  # 显示字符
plt.show()
```
这样就可以生成pos*d_model 形状的pos_embed 
和word_embed 的形状相同，直接concat就行
#### 相对编码
[参考博客](https://blog.csdn.net/python123456_/article/details/141352984)
相对编码是不直接用position_embedding表示
而是在后续计算Attention-score的过程中
而是将位置编码作为一个偏置项带入到attention-score中去
从而影响最后的结果

#### 旋转编码 RoPE
这其实也是一种相对编码，只不过用的旋转关系，而不是一个偏置项，来表示位置关系的影响，这里我不深入了解，知道概念即可


### 2.Multihead-Attention
[参考博客](https://blog.csdn.net/xiaoh_7/article/details/140019530)
多头注意力
主要的过程在草稿纸上演算了一遍
清楚了代码每一步在干什么

### 3.Feed Forward
FFN=fc+relu+fc
第一个fc用来升维,将原有的小维度提升，可以展示出更多的维度之间的关系
而relu则用来将刚刚产生的，更细节的输入来进行activation
再用fc降维回到一开始的维度
FFN work的机制:我的理解就是更细节的activation，比起直接relu多了升降维度的过程，能筛出更多细节
### 4.Add & Norm
### 5.Masked-Multihead-Attention

### 6.FC
### 7.softmax

完成了第一次复现
理解了各个模块的构成，以及transformer的原理
接下来会经常复现transformer的


# 一些深入一些的思考
1.为什么是LayerNorm? LayerNorm和BatchNorm区别在哪里？以及GroupNorm区别？
2.有几种编码方式？ 说一说相对编码和旋转编码RoPE
3.Encoder-Only结构和Decoder-Only结构的代表模型和处理任务是？
4.Mask操作怎么做的，说一说？



之后进入到diffusion的学习

