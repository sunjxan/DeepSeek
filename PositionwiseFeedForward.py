import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Transformer的前馈神经网络（FFN）模块。
        
        Args:
            d_model (int): 输入和输出的特征维度（即词嵌入的维度）。
            d_ff (int): 中间层的维度（通常比d_model大）。
            dropout (float): Dropout概率，默认为0.1。
        """
        super().__init__()
        # 第一个线性层：将输入从d_model扩展到d_ff维度
        self.linear1 = nn.Linear(d_model, d_ff)
        # 第二个线性层：将中间层从d_ff恢复回d_model维度
        self.linear2 = nn.Linear(d_ff, d_model)
        # 并行线性变换层：用于门控计算
        self.linear3 = nn.Linear(d_model, d_ff)
        # 激活函数
        self.activation = nn.SiLU()
        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入序列，shape: (batch_size, seq_len, d_model)
        
        Returns:
            torch.Tensor: 输出序列，shape: (batch_size, seq_len, d_model)
        """
        # 输入x的shape: (batch_size, seq_len, d_model)
        
        # 第一步：通过第一个线性层（扩展维度）
        y = self.linear1(x)
        z = self.linear3(x)
        # 此时x的shape: (batch_size, seq_len, d_ff)
        
        # 第二步：应用激活函数
        y = self.activation(y)
        # 激活函数不改变shape，仍为(batch_size, seq_len, d_ff)
        
        # 第三步：应用Dropout
        x = self.dropout(y * z)
        # Dropout不改变shape，仍为(batch_size, seq_len, d_ff)
        
        # 第四步：通过第二个线性层（恢复维度）
        x = self.linear2(x)
        # 此时x的shape恢复为: (batch_size, seq_len, d_model)
        
        return x
