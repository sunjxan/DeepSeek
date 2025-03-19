import torch
import torch.nn as nn

from PositionwiseFeedForward import PositionwiseFeedForward

class Gate(nn.Module):
    """
    混合专家模型（MoE）中的门控路由机制
    
    参数:
        dim (int): 输入特征的维度
        topk (int): 每个输入激活的top专家数
        n_groups (int): 路由分组数
        topk_groups (int): 输入路由的目标组数
        score_func (str): 评分函数（'softmax' 或 'sigmoid'）
        route_scale (float): 路由权重的缩放因子
        weight (torch.nn.Parameter): 可学习的门控权重
        bias (Optional[torch.nn.Parameter]): 可选的门控偏置项
    """
    def __init__(self, args):
        """
        初始化门控模块

        参数:
            args: 包含模型配置参数的ModelArgs对象
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts  # 每个样本激活的专家数
        self.n_groups = args.n_expert_groups  # 专家分组数量
        self.topk_groups = args.n_limited_groups  # 每个样本选择的分组数
        self.route_scale = args.route_scale  # 路由权重缩放因子
        
        # 路由权重矩阵：形状为(专家数量, 特征维度)
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        # 特殊情况下（维度为7168）添加偏置项
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x):
        """
        前向传播过程：
        1. 计算路由分数
        2. 分组选择与掩码处理
        3. 选择topk专家并计算路由权重

        参数:
            x: 输入张量，形状为(batch_size, dim)

        返回:
            (weights, indices): 路由权重和专家索引
        """
        # 计算原始路由分数 [batch_size, n_experts]
        scores = torch.matmul(x, self.weight.transpose(-2, -1))
        scores = scores.softmax(dim=-1)  # 标准化为概率分布
        original_scores = scores  # 保存原始分数用于后续计算
        
        # 添加偏置项（如果存在）
        if self.bias is not None:
            scores = scores + self.bias
        
        # 分组路由逻辑
        if self.n_groups > 1:
            # 将分数重塑为 [batch_size, n_groups, group_size]
            scores = scores.view(x.size(0), self.n_groups, -1)
            
            # 计算每个组的得分：当没有偏置时取最大值，有偏置时取top2的和
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            
            # 选择得分最高的topk_groups个组
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            
            # 创建掩码：将未选中的组标记为-inf
            mask = torch.ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf")).flatten(1)
        
        # 选择topk专家
        indices = torch.topk(scores, self.topk, dim=-1)[1]  # [batch_size, topk]
        
        # 从原始分数中收集路由权重，并应用缩放因子
        weights = original_scores.gather(1, indices)  # [batch_size, topk]
        weights *= self.route_scale  # 应用路由缩放
        
        return weights, indices  # 保持数据类型一致

class MoE(nn.Module):
    """
    混合专家模型（Mixture of Experts）模块
    
    参数:
        dim (int): 输入特征的维度
        n_routed_experts (int): 模型中的专家总数
        n_local_experts (int): 分布式系统中本地处理的专家数
        n_activated_experts (int): 每个输入激活的专家数
        gate (nn.Module): 路由输入到专家的门控机制
        experts (nn.ModuleList): 专家模块列表
        shared_experts (nn.Module): 应用于所有输入的共享专家
    """
    def __init__(self, args):
        """
        初始化MoE模块

        参数:
            args (ModelArgs): 包含MoE参数的模型参数
        """
        super().__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts  # 总专家数量
        self.n_activated_experts = args.n_activated_experts  # 激活的专家数
        
        # 初始化门控模块
        self.gate = Gate(args)
        
        # 初始化专家列表（使用MLP作为专家网络）
        self.experts = nn.ModuleList([
            PositionwiseFeedForward(args.dim, args.moe_inter_dim) for i in range(self.n_routed_experts)
        ])
        
        # 共享专家处理所有输入
        self.shared_experts = PositionwiseFeedForward(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x):
        """
        前向传播过程：
        1. 通过门控获取路由权重和专家索引
        2. 稀疏激活选中的专家进行计算
        3. 叠加共享专家的计算结果

        参数:
            x: 输入张量，形状为(batch_size, seq_len, dim)

        返回:
            输出张量，形状与输入相同
        """
        original_shape = x.size()
        x = x.view(-1, self.dim)  # 展平为二维张量 [batch*seq_len, dim]
        
        # 获取路由权重和专家索引
        weights, indices = self.gate(x)  # weights: [batch*seq_len, topk], indices: [batch*seq_len, topk]
        
        # 初始化输出张量
        y = torch.zeros_like(x)
        
        # 统计每个专家被选中的次数（用于负载均衡）
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        
        # 遍历本地专家进行计算
        for i in range(self.n_routed_experts):
            if counts[i] == 0:  # 跳过未被选中的专家
                continue
                
            expert = self.experts[i]  # 获取对应的专家模块
            idx, top_pos = torch.where(indices == i)  # 找到选择该专家的样本索引
            
            # 加权计算结果并累加到输出
            y[idx] += expert(x[idx]) * weights[idx, top_pos, None]  # [n_samples, dim]
        
        # 共享专家处理所有输入
        z = self.shared_experts(x)  # [batch*seq_len, dim]
        
        # 合并结果并恢复原始形状
        return (y + z).view(original_shape)
