import math
import torch
import torch.nn as nn

def apply_rotary_emb(x, freqs_cis):
    """
    应用旋转位置编码到输入张量
    
    实现步骤：
    1. 将输入转换为复数形式
    2. 与预计算的频率复数相乘（相当于旋转向量）
    3. 转换回实数形式
    
    参数:
        x (torch.Tensor): 输入张量，形状为(..., head_dim)
        freqs_cis (torch.Tensor): 预计算的复数频率矩阵
    
    返回:
        torch.Tensor: 旋转后的张量，保持原始形状
    """
    dtype = x.dtype  # 保存原始数据类型
    # 将输入转换为复数形式（假设最后两维可以分成实部和虚部）
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    # 调整频率矩阵形状以匹配输入（添加广播维度）
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    # 复数乘法实现旋转，然后转换回实数
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)  # 恢复原始数据类型

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
        
        return output

class MultiHeadLatentAttention(nn.Module):
    """
    多头注意力层（Multi-head Latent Attention），包含LoRA支持和KV缓存
    
    结构特点：
    - 可选LoRA低秩适应：当q_lora_rank>0时启用查询LoRA
    - 分离的位置编码处理：将Q分为带位置编码和不带位置编码部分
    - 动态KV缓存管理：缓存历史K/V值用于自回归生成
    
    参数:
        dim (int): 输入特征的维度
        n_heads (int): 注意力头数
        n_local_heads (int): 分布式系统中本地注意力头数
        q_lora_rank (int): 低秩查询投影的秩
        kv_lora_rank (int): 低秩键/值投影的秩
        qk_nope_head_dim (int): 非位置敏感查询/键投影的维度
        qk_rope_head_dim (int): 旋转位置敏感查询/键投影的维度
        qk_head_dim (int): 查询/键投影的总维度
        v_head_dim (int): 值投影的维度
        softmax_scale (float): 注意力计算中softmax的缩放因子
    """
    def __init__(self, args, dropout=0.1):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads        # 总头数
        
        # LoRA配置
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        
        # 头维度分解
        self.qk_nope_head_dim = args.qk_nope_head_dim  # 无位置编码部分
        self.qk_rope_head_dim = args.qk_rope_head_dim  # 带位置编码部分
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim  # Q/K总维度
        self.v_head_dim = args.v_head_dim             # V头维度
        
        # 查询投影（可选LoRA）
        if self.q_lora_rank == 0:  # 标准线性变换
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim)
        else:  # LoRA分解：W = W_a * W_b
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank)
            self.q_norm = nn.RMSNorm(self.q_lora_rank)  # 归一化层
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        
        # 键值投影（统一处理）
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank)  # 归一化
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        
        # 输出投影
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim)
        
        # 注意力缩放因子
        softmax_scale = self.qk_head_dim ** -0.5  # 1/sqrt(d_k)
        # 长上下文缩放调整
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            softmax_scale *= mscale ** 2  # 平方缩放
        
        # 注册KV缓存（非持久化缓冲区）
        self.register_buffer("k_cache", torch.zeros(
            args.max_batch_size, args.max_seq_len, self.n_heads, self.qk_head_dim
        ), persistent=False)
        self.register_buffer("v_cache", torch.zeros(
            args.max_batch_size, args.max_seq_len, self.n_heads, self.v_head_dim
        ), persistent=False)
        
        self.attn = ScaledDotProductAttention(softmax_scale, dropout)
    
    def forward(self, x, start_pos, freqs_cis, mask=None):
        """
        前向传播过程
        
        处理步骤：
        1. 投影得到查询Q，分解为无位置编码和有位置编码部分
        2. 投影得到键K和值V，分离位置编码部分
        3. 更新KV缓存
        4. 计算注意力分数
        5. 应用mask（训练时）和softmax
        6. 聚合价值信息并投影输出
        
        参数:
            x: 输入张量 (batch_size, seq_len, dim)
            start_pos: 当前输入的起始位置（用于缓存）
            freqs_cis: 预计算的旋转位置编码
            mask: 注意力mask（防止看到未来信息）
        
        返回:
            输出张量，形状与输入相同
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        
        # 查询投影（带可选LoRA）
        if self.q_lora_rank == 0:
            q = self.wq(x)  # (bsz, seqlen, n_heads*qk_head_dim)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        
        # 分解Q为无位置编码和有位置编码部分
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)  # 应用旋转位置编码
        
        # 键值投影
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)  # 添加头维度后应用RoPE
        
        # 合并Q的两个部分
        q = torch.cat([q_nope, q_pe], dim=-1)
        
        # 处理键值投影
        kv = self.wkv_b(self.kv_norm(kv))
        kv = kv.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        
        # 合并K的两个部分并扩展头维度
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)
        
        # 更新KV缓存
        with torch.no_grad():
            # 克隆原有缓存以避免原地操作
            new_k_cache = self.k_cache.clone()
            new_v_cache = self.v_cache.clone()
            
            # 更新克隆后的副本
            new_k_cache[:bsz, start_pos:end_pos] = k.detach()
            new_v_cache[:bsz, start_pos:end_pos] = v.detach()
            
            # 替换原缓存
            self.k_cache = new_k_cache
            self.v_cache = new_v_cache
        
        # 应用掩码（如果存在）
        if mask is not None:
            # 扩展掩码维度以匹配多头 (batch_size, seq_len, 1, seq_len) -> 广播到num_heads
            mask = mask.unsqueeze(2)
        
        output = self.attn(q, self.k_cache[:bsz, :end_pos], self.v_cache[:bsz, :end_pos], mask)
        
        # 投影回模型维度
        return self.wo(output.flatten(2))
