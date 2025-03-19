class ModelArgs:
    """
    模型参数配置类，用于定义模型超参数和结构参数

    参数:
        max_batch_size (int): 最大批处理大小，决定缓存分配
        max_seq_len (int): 最大序列长度，用于位置编码和缓存
        dtype (Literal["bf16", "fp8"]): 计算数据类型（当前被注释）
        vocab_size (int): 词表大小，决定嵌入层维度
        dim (int): 模型隐藏层维度
        inter_dim (int): MLP中间层维度
        moe_inter_dim (int): MoE专家网络的中间层维度
        n_layers (int): 模型总层数（Transformer块数量）
        n_dense_layers (int): 使用普通MLP的层数（前n层使用MLP，后续层使用MoE）
        n_heads (int): 注意力头的数量
        # MoE相关参数
        n_routed_experts (int): 路由专家的总数
        n_shared_experts (int): 共享专家数量（所有输入都会经过的专家）
        n_activated_experts (int): 每个token实际使用的专家数量
        n_expert_groups (int): 专家分组数量（用于分布式计算）
        n_limited_groups (int): 每个token可以选择的最大专家组数量
        score_func (Literal["softmax", "sigmoid"]): 路由分数计算方式（当前被注释）
        route_scale (float): 路由权重的缩放因子
        # 注意力相关参数
        q_lora_rank (int): 查询矩阵的LoRA降维秩（0表示不使用LoRA）
        kv_lora_rank (int): 键值矩阵的LoRA降维秩
        qk_nope_head_dim (int): 无位置编码的Q/K头维度
        qk_rope_head_dim (int): 带旋转位置编码的Q/K头维度
        v_head_dim (int): 值向量的头维度
        # YARN扩展相关参数
        original_seq_len (int): 原始训练的序列长度
        rope_theta (float): 旋转位置编码的基础频率
        rope_factor (float): 序列长度扩展因子
        beta_fast (int): 快速beta调整参数
        beta_slow (int): 慢速beta调整参数
        mscale (float): 注意力缩放因子
    """
    max_batch_size: int = 32         # 根据GPU显存调整
    max_seq_len: int = 100           # 支持最大tokens数
    # dtype: Literal["bf16", "fp8"] = "bf16"
    # vocab_size: int = 10240          # 根据词表实际大小设置
    dim: int = 200                   # 隐藏层维度
    inter_dim: int = 300             # MLP中间层扩展维度
    moe_inter_dim: int = 250         # MoE专家中间层维度
    n_layers: int = 6                # 模型总层数
    n_dense_layers: int = 1          # 前1层使用普通MLP
    n_heads: int = 16                # 注意力头数
    
    # MoE参数配置
    n_routed_experts: int = 64       # 总专家数量
    n_shared_experts: int = 2        # 共享专家数量
    n_activated_experts: int = 6     # 每个token激活的专家数
    n_expert_groups: int = 1         # 专家分组（分布式训练用）
    n_limited_groups: int = 1        # 每个token可选的最大组数
    # score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.          # 路由权重缩放因子
    
    # 注意力机制参数
    q_lora_rank: int = 0             # 查询LoRA秩（0表示禁用）
    kv_lora_rank: int = 512          # 键值LoRA秩
    qk_nope_head_dim: int = 128      # 无位置编码头维度
    qk_rope_head_dim: int = 64       # 旋转位置编码头维度
    v_head_dim: int = 128            # 值向量头维度
    
    # YARN扩展参数（支持长上下文）
    original_seq_len: int = 4096     # 基础训练长度
    rope_theta: float = 10000.0      # RoPE基础频率
    rope_factor: float = 40          # 长度扩展因子
    beta_fast: int = 32              # 快速调整参数
    beta_slow: int = 1               # 慢速调整参数
    mscale: float = 1.               # 注意力缩放因子