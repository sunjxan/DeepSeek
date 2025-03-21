import math
import torch
import torch.nn as nn

from Decoder import Decoder

def precompute_freqs_cis(args):
    """
    预计算旋转位置编码的复数形式频率矩阵
    
    算法流程：
    1. 根据基础参数计算初始频率
    2. 如果序列长度超过原始长度，进行长度外推修正
    3. 生成所有位置的频率矩阵
    4. 转换为复数形式（cos + i*sin）
    
    参数:
        args (ModelArgs): 包含所有位置编码参数的模型参数
        
    返回:
        torch.Tensor: 预计算好的复数频率矩阵，形状为(seq_len, dim//2)
    """
    dim = args.qk_rope_head_dim       # 使用带位置编码的头维度
    seqlen = args.max_seq_len         # 最大序列长度
    beta_fast = args.beta_fast        # 长度外推参数
    beta_slow = args.beta_slow
    base = args.rope_theta            # 10000
    factor = args.rope_factor         # 扩展因子

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        计算旋转位置嵌入中给定旋转次数所需的校正维度。（用于长度外推）
        
        公式推导：
        d_c = dim * ln(max_seq_len / (num_rotations * 2π)) / (2 ln(base))
        
        参数:
            num_rotations (float): 需要计算校正的旋转次数
            dim (int): 嵌入空间的维度
            base (float): 指数计算的基础值
            max_seq_len (int): 最大序列长度
        
        返回:
            float: 基于输入参数的校正维度
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        计算旋转位置嵌入的校正维度范围。（生成平滑过渡区域）
        
        参数:
            low_rot (float): 旋转次数的下限
            high_rot (float): 旋转次数的上限
            dim (int): 嵌入空间的维度
            base (float): 指数计算的基础值
            max_seq_len (int): 最大序列长度
        
        返回:
            Tuple[int, int]: 校正维度的范围（下限，上限），已截断到有效索引范围内
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        生成线性斜坡掩码，用于平滑混合基础频率和修正频率
        
        参数:
            min (float): 斜坡函数的最小值
            max (float): 斜坡函数的最大值
            dim (int): 斜坡张量的维度
        
        返回:
            torch.Tensor: 形状为 (dim,) 的张量，值在0到1之间线性插值，并截断到[0,1]范围
        """
        if min == max:  # 避免除零
            max += 0.001
        # 生成0-1的线性插值，然后裁剪到[0,1]区间
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        return torch.clamp(linear_func, 0, 1)

    # 基础频率计算：1/(base^(2i/dim))，i从0到dim//2-1
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    
    # 长度外推处理（当实际长度超过原始训练长度时）
    if seqlen > args.original_seq_len:
        # 计算需要调整的维度范围
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        # 生成平滑过渡掩码（中间区域混合两种频率）
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        # 混合基础频率和调整后的频率
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    # 生成位置-频率矩阵
    t = torch.arange(seqlen)  # 所有位置索引
    freqs = torch.outer(t, freqs)  # 外积得到(seqlen, dim//2)矩阵
    # 转换为复数形式：e^(iθ) = cosθ + i sinθ
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

class Transformer(nn.Module):
    def __init__(self, args, dropout=0.1):
        """
        Transformer 模型
        Args:
            vocab_size (int): 词表大小
            d_model (int): 模型维度（输入/输出维度）
            num_heads (int): 多头注意力头数
            num_layers (int): Decoder 层数
            d_ff (int): 前馈网络中间层维度
            max_seq_len (int): 最大序列长度
            dropout (float): Dropout 概率
        """
        super().__init__()
        self.max_seq_len = args.max_seq_len
        
        # 1. 词嵌入层
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        
        # 2. 解码器
        self.decoder = Decoder(args, dropout)
        
        # 3. 最终线性层
        self.generator = nn.Linear(args.dim, args.vocab_size)
        
        # 权重绑定：输入嵌入和输出层共享权重
        self.embed.weight = self.generator.weight
        
        # 注册预计算的位置编码（非持久化缓冲区）
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)
    
    def forward(self, input_ids, start_pos=0, mask=None):
        """
        前向传播
        Args:
            input_ids (Tensor): 源序列 (batch_size, seq_len)
            mask (Tensor): 序列掩码 (batch_size, seq_len, seq_len)
        Returns:
            output (Tensor): 输出概率分布 (batch_size, seq_len, vocab_size)
        """
        seq_len = input_ids.size(-1)
        assert seq_len <= self.max_seq_len, f"序列长度{seq_len}超过最大限制{self.max_seq_len}"
        
        # 1. 词嵌入
        emb = self.embed(input_ids)  # (batch_size, seq_len, d_model)
        
        # 获取当前位置编码
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]

        # 2. 解码器处理
        decoder_output = self.decoder(emb, start_pos, freqs_cis, mask)  # (batch_size, seq_len, d_model)
        
        # 3. 输出层映射到词表
        output = self.generator(decoder_output)  # (batch_size, seq_len, vocab_size)
        
        return output
    
    def init_parameters(self, init_type='xavier'):
        """
        初始化模型参数
        Args:
            init_type (str): 初始化类型，可选 'xavier'（默认）或 'kaiming'
        """
        for name, param in self.named_parameters():
            if param.dim() > 1:  # 仅初始化矩阵权重，忽略偏置和RMSNorm参数
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name:  # 偏置初始化为零
                nn.init.zeros_(param)
            # RMSNorm参数保持默认初始化（gamma=1, beta=0）
    
    @staticmethod
    def generate_padding_mask(seq, pad_id=0):
        """生成填充掩码（pad位置为False）"""
        return (seq != pad_id).unsqueeze(-2)  # (batch_size, 1, seq_len)
    
    @staticmethod
    def generate_causal_mask(seq_len):
        """生成因果掩码（下三角为True）"""
        return torch.tril(torch.ones(seq_len, seq_len)) == 1  # (seq_len, seq_len)
    
    @classmethod
    def generate_mask(cls, seq, pad_id=0):
        '''结合填充掩码和因果掩码得到目标序列掩码'''
        return cls.generate_padding_mask(seq, pad_id) & cls.generate_causal_mask(seq.size(-1)).to(seq.device)   # (batch_size, seq_len, seq_len)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
