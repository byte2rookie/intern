# 1.Diffusion的原理
DDPM 
利用去噪算法来一步步将文字信息和norm噪音输入 一步步还原成最后的图片
DALL.E Imagen等等文生图技术的套路 和stablediffusion一致
都是经过文字->text-encoder->去噪->image-decoder 生成的图片

## 生成图像的评价指标
### FID
FID就是 将生成图片和真实图片都放入到提前准备好的CNN网络进行判断，得到两个图片集的分布，用他们分布之间的距离来表示生成图像和真实图像之间的相似度
### CLIP-score
CLIPs计算的方式就是将文字输入和图片输入进行encoder操作
得到文字向量和图片向量
CLIPs用公式计算这两个向量的相关度
只要相关度高，CLIPs就高，否则就低

## 怎么训练decoder
### auto-encoder
输入->encoder->中间产物->decoder->输出  让输入和输出的结果越接近越好
得到的encoder和decoder就是我们所需要的decoder

## Maximal Likelihood Esitimation
