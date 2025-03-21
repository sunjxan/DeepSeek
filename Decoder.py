import torch
import torch.nn as nn

from MultiHeadLatentAttention import MultiHeadLatentAttention
from PositionwiseFeedForward import PositionwiseFeedForward
from MoE import MoE
from SublayerConnection import SublayerConnection

class DecoderLayer(nn.Module):
    def __init__(self, layer_id, args, dropout=0.1):
        """
        Transformer的单个解码器层。
        
        Args:
            d_model (int): 输入的特征维度（即词嵌入的维度）。
            num_heads (int): 多头注意力机制的头数。
            d_ff (int): 前馈网络中间层的维度。
            dropout (float): Dropout概率，默认为0.1。
        """
        super().__init__()
        
        # 1. 带掩码的多头自注意力层（用于处理目标序列）
        self.self_attn = MultiHeadLatentAttention(args, dropout)
        
        # 2. 前馈网络
        self.ffn = PositionwiseFeedForward(
            args.dim, args.inter_dim, dropout) if layer_id < args.n_dense_layers else MoE(args)
        
        # 3. 层归一化（RMSNorm） + Dropout层
        self.sublayer1 = SublayerConnection(args.dim, dropout)
        self.sublayer2 = SublayerConnection(args.dim, dropout)
    
    def forward(self, x, start_pos, freqs_cis, mask=None):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 目标序列输入，shape: (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): 目标序列掩码，shape: (batch_size, seq_len, seq_len)
        
        Returns:
            torch.Tensor: 解码器层输出，shape: (batch_size, seq_len, d_model)
        """
        # ----------------- 步骤1：带掩码的自注意力 -----------------
        # 输入x的shape: (batch_size, seq_len, d_model)
        x = self.sublayer1(x, lambda x: self.self_attn(x, start_pos, freqs_cis, mask))
        
        # ----------------- 步骤3：前馈网络 -----------------
        x = self.sublayer2(x, self.ffn)
        
        return x  # 输出shape: (batch_size, seq_len, d_model)

class Decoder(nn.Module):
    def __init__(self, args, dropout=0.1):
        """
        Transformer Decoder 模块
        Args:
            num_layers (int): 解码器层数。
            d_model (int): 输入的特征维度。
            num_heads (int): 多头注意力的头数。
            d_ff (int): 前馈网络中间层维度。
            dropout (float): Dropout概率，默认为0.1。
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            DecoderLayer(layer_id, args, dropout) for layer_id in range(args.n_layers)
        ])
        
        self.norm = nn.RMSNorm(args.dim)  # 最终归一化层（SublayerConnection选用Post-LN 结构时删除）
    
    def forward(self, x, start_pos, freqs_cis, mask=None):
        """
        前向传播
        Args:
            x (torch.Tensor): 目标序列输入，shape: (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): 目标序列掩码，shape: (batch_size, seq_len, seq_len)
        
        Returns:
            torch.Tensor: 解码器输出，shape: (batch_size, seq_len, d_model)
        """
        # 逐层传递输入
        for layer in self.layers:
            x = layer(x, start_pos, freqs_cis, mask)
        
        x = self.norm(x)  # 最终归一化
        
        return x
