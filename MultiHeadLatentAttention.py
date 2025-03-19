import math
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale, dropout=0.1):
        """
        缩放点积注意力机制
        Args:
            dropout (float): Dropout概率，默认为0.1
        """
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(dropout)  # Dropout层
    
    def forward(self, Q, K, V, mask=None):
        """
        前向传播
        Args:
            Q: 查询张量, shape (batch_size, seq_len_q, num_heads, d_k)
            K: 键张量, shape (batch_size, seq_len_k, num_heads, d_k)
            V: 值张量, shape (batch_size, seq_len_k, num_heads, d_v)
            mask: 掩码张量, shape (batch_size, 1, seq_len_q, seq_len_k)
        
        Returns:
            注意力输出: shape (batch_size, seq_len_q, num_heads, d_v)
            注意力权重: shape (batch_size, seq_len_q, num_heads, seq_len_k)
        """
        # 计算注意力分数 Q@K^T / sqrt(d_k)
        scores = torch.einsum("bshd,bthd->bsht", Q, K) * self.scale
        # scores shape: (batch_size, seq_len_q, num_heads, seq_len_k)
        
        # 应用掩码（如果需要）
        if mask is not None:
            # 将mask中为False的位置替换为负无穷大（softmax后趋近于0）
            scores = scores.masked_fill(mask == 0, float('-inf'))
            # mask需要能广播到scores的形状
            # src attention，mask形状(1, S)，广播后(S, S)，右侧为False，用于重新编码时忽略pad
            # tgt attention，mask形状(T, T)，右侧和右上方为False，用于重新编码时忽略pad和该词后面的词
            # tgt-src attention，mask形状(1, S)，广播后(T, S)，右侧为False，每个词根据src的value重新编码，可用于预测下一个词
        
        # 计算注意力权重（最后一维进行softmax）
        attn_weights = torch.softmax(scores, dim=-1)
        # attn_weights shape: (batch_size, seq_len_q, num_heads, seq_len_k)
        
        attn_weights = self.dropout(attn_weights)
        
        # 注意力加权求和
        output = torch.einsum("bsht,bthd->bshd", attn_weights, V)
        # output shape: (batch_size, seq_len_q, num_heads, d_v)
        
        return output, attn_weights

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        多头注意力机制
        Args:
            d_model: 输入维度（总维度）
            num_heads: 注意力头的数量
            dropout: dropout概率
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 线性变换矩阵（无偏置）
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)  # (d_model, 3*d_model) 合并QKV投影
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # (d_model, d_model)
        
        self.attn = ScaledDotProductAttention(dropout)
    
    def forward(self, x, mask=None):
        """
        前向传播
        Args:
            x: QKV向量 (batch_size, seq_len, d_model)
            mask: 掩码 (batch_size, seq_len, seq_len)
        Returns:
            输出: (batch_size, seq_len, d_model)
            注意力权重: (batch_size, seq_len, num_heads, seq_len)
        """    
        batch_size = x.size(0)
        
        # 线性变换 + 分割多头
        QKV = self.W_qkv(x)  # (batch_size, seq_len, 3*d_model)
        Q, K, V = QKV.chunk(3, dim=-1)  # 各(batch_size, seq_len, d_model)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k)  # (batch_size, seq_len, num_heads, d_k)
        K = K.view(batch_size, -1, self.num_heads, self.d_k)  # (batch_size, seq_len, num_heads, d_k)
        V = V.view(batch_size, -1, self.num_heads, self.d_k)  # (batch_size, seq_len, num_heads, d_k)
        
        # 应用掩码（如果存在）
        if mask is not None:
            # 扩展掩码维度以匹配多头 (batch_size, seq_len, 1, seq_len) -> 广播到num_heads
            mask = mask.unsqueeze(2)
        
        output, attn_weights = self.attn(Q, K, V, mask)
        
        # 拼接所有头 (batch_size, seq_len, d_model)
        output = output.contiguous()
        output = output.view(batch_size, -1, self.d_model)
        
        # 最终线性变换
        output = self.W_o(output)
        
        return output
