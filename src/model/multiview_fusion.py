"""
多视图融合模块

提供多种策略将8个视图的LLaVA特征融合为统一表示。

融合策略:
1. MeanFusion: 简单平均池化
2. ConcatFusion: 拼接后投影
3. AttentionFusion: 注意力加权融合
4. PerceiverFusion: Perceiver风格的可学习查询融合 (推荐)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MeanFusion(nn.Module):
    """
    最简单的融合方式：对所有视图特征取平均

    优点: 计算量小，无额外参数
    缺点: 无法学习视图重要性
    """
    def __init__(self, hidden_dim: int = 4096):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: [B, N_views, Seq_Len, Hidden_Dim] 多视图特征
            mask: [B, N_views] 视图有效掩码 (可选)

        Returns:
            fused: [B, Seq_Len, Hidden_Dim] 融合后的特征
        """
        if mask is not None:
            # 带掩码的加权平均
            mask = mask.unsqueeze(-1).unsqueeze(-1)  # [B, N_views, 1, 1]
            features = features * mask
            fused = features.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        else:
            fused = features.mean(dim=1)
        return fused


class ConcatFusion(nn.Module):
    """
    拼接融合：将所有视图特征拼接后通过MLP压缩

    优点: 保留所有视图信息
    缺点: 参数量随视图数线性增长
    """
    def __init__(self, hidden_dim: int = 4096, n_views: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_views = n_views

        # 视图位置编码
        self.view_embed = nn.Embedding(n_views, hidden_dim)

        # 压缩投影
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * n_views, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: [B, N_views, Seq_Len, Hidden_Dim]

        Returns:
            fused: [B, Seq_Len, Hidden_Dim]
        """
        B, N, S, D = features.shape

        # 添加视图位置编码
        view_ids = torch.arange(N, device=features.device)
        view_emb = self.view_embed(view_ids)  # [N, D]
        features = features + view_emb.view(1, N, 1, D)

        # 拼接所有视图
        features = features.permute(0, 2, 1, 3)  # [B, S, N, D]
        features = features.reshape(B, S, N * D)  # [B, S, N*D]

        # 压缩投影
        fused = self.projection(features)  # [B, S, D]
        return fused


class AttentionFusion(nn.Module):
    """
    注意力融合：学习每个视图的重要性权重

    优点: 自适应选择重要视图
    缺点: 每个位置使用相同的视图权重
    """
    def __init__(self, hidden_dim: int = 4096, n_views: int = 8, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_views = n_views
        self.n_heads = n_heads

        # 视图位置编码
        self.view_embed = nn.Embedding(n_views, hidden_dim)

        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: [B, N_views, Seq_Len, Hidden_Dim]
            mask: [B, N_views] 视图有效掩码

        Returns:
            fused: [B, Seq_Len, Hidden_Dim]
        """
        B, N, S, D = features.shape

        # 添加视图位置编码
        view_ids = torch.arange(N, device=features.device)
        view_emb = self.view_embed(view_ids)
        features = features + view_emb.view(1, N, 1, D)

        # 重组为 [B*S, N, D] 以便注意力计算
        features = features.permute(0, 2, 1, 3).reshape(B * S, N, D)

        # 使用第一个视图作为query (或可学习query)
        query = features[:, 0:1, :]  # [B*S, 1, D]

        # 自注意力融合
        if mask is not None:
            key_padding_mask = ~mask.unsqueeze(1).expand(-1, S, -1).reshape(B * S, N)
        else:
            key_padding_mask = None

        fused, _ = self.attention(query, features, features, key_padding_mask=key_padding_mask)
        fused = fused.squeeze(1).reshape(B, S, D)

        fused = self.output_proj(fused)
        return fused


class PerceiverFusion(nn.Module):
    """
    Perceiver风格融合：使用可学习的latent query通过cross-attention聚合多视图信息

    参考: Perceiver: General Perception with Iterative Attention (ICML 2021)

    优点:
    - 独立于视图数量的固定计算量
    - 可学习的查询能够捕获CAD重建所需的关键信息
    - 支持任意数量的视图输入

    缺点:
    - 需要调整latent数量
    """
    def __init__(
        self,
        hidden_dim: int = 4096,
        n_views: int = 8,
        n_latents: int = 64,  # 可学习查询数量
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_views = n_views
        self.n_latents = n_latents

        # 视图位置编码
        self.view_embed = nn.Embedding(n_views, hidden_dim)

        # 可学习的latent查询
        self.latent_queries = nn.Parameter(torch.randn(n_latents, hidden_dim) * 0.02)

        # Cross-attention层 (latent queries attend to multi-view features)
        self.cross_attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=n_heads,
                    dropout=dropout,
                    batch_first=True
                ),
                'cross_norm': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                ),
                'ffn_norm': nn.LayerNorm(hidden_dim),
            })
            for _ in range(n_layers)
        ])

        # 输出投影 (从n_latents到seq_len)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: [B, N_views, Seq_Len, Hidden_Dim]
            mask: [B, N_views] 视图有效掩码

        Returns:
            fused: [B, n_latents, Hidden_Dim] 融合后的latent表示
        """
        B, N, S, D = features.shape

        # 添加视图位置编码
        view_ids = torch.arange(N, device=features.device)
        view_emb = self.view_embed(view_ids)
        features = features + view_emb.view(1, N, 1, D)

        # 展平多视图特征: [B, N*S, D]
        kv = features.reshape(B, N * S, D)

        # 初始化latent查询: [B, n_latents, D]
        latents = self.latent_queries.unsqueeze(0).expand(B, -1, -1)

        # 构建key_padding_mask
        if mask is not None:
            # mask: [B, N] -> [B, N*S]
            key_padding_mask = ~mask.unsqueeze(-1).expand(-1, -1, S).reshape(B, N * S)
        else:
            key_padding_mask = None

        # 多层cross-attention
        for layer in self.cross_attention_layers:
            # Cross-attention
            residual = latents
            latents_attended, _ = layer['cross_attn'](
                latents, kv, kv,
                key_padding_mask=key_padding_mask
            )
            latents = layer['cross_norm'](residual + latents_attended)

            # FFN
            residual = latents
            latents = layer['ffn_norm'](residual + layer['ffn'](latents))

        # 输出投影
        fused = self.output_proj(latents)  # [B, n_latents, D]

        return fused


class MultiViewFusion(nn.Module):
    """
    多视图融合模块的统一接口

    支持多种融合策略:
    - 'mean': 简单平均
    - 'concat': 拼接投影
    - 'attention': 注意力融合
    - 'perceiver': Perceiver风格融合 (推荐)
    """
    def __init__(
        self,
        hidden_dim: int = 4096,
        n_views: int = 8,
        fusion_type: str = 'perceiver',
        n_latents: int = 64,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.n_latents = n_latents

        if fusion_type == 'mean':
            self.fusion = MeanFusion(hidden_dim)
        elif fusion_type == 'concat':
            self.fusion = ConcatFusion(hidden_dim, n_views, dropout)
        elif fusion_type == 'attention':
            self.fusion = AttentionFusion(hidden_dim, n_views, n_heads, dropout)
        elif fusion_type == 'perceiver':
            self.fusion = PerceiverFusion(
                hidden_dim, n_views, n_latents, n_heads, n_layers, dropout
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: [B, N_views, Seq_Len, Hidden_Dim]
            mask: [B, N_views] 视图有效掩码

        Returns:
            对于 mean/concat/attention: [B, Seq_Len, Hidden_Dim]
            对于 perceiver: [B, n_latents, Hidden_Dim]
        """
        return self.fusion(features, mask)


# === 测试代码 ===
if __name__ == '__main__':
    print("=== 多视图融合模块测试 ===\n")

    B, N_views, Seq_Len, Hidden_Dim = 2, 8, 128, 4096
    features = torch.randn(B, N_views, Seq_Len, Hidden_Dim)
    mask = torch.ones(B, N_views).bool()
    mask[0, 6:] = False  # 模拟部分视图无效

    print(f"输入形状: {features.shape}")
    print(f"掩码形状: {mask.shape}\n")

    for fusion_type in ['mean', 'concat', 'attention', 'perceiver']:
        print(f"--- {fusion_type.upper()} 融合 ---")

        if fusion_type == 'perceiver':
            module = MultiViewFusion(Hidden_Dim, N_views, fusion_type, n_latents=64)
        else:
            module = MultiViewFusion(Hidden_Dim, N_views, fusion_type)

        output = module(features, mask)
        n_params = sum(p.numel() for p in module.parameters())

        print(f"  输出形状: {output.shape}")
        print(f"  参数量: {n_params:,}\n")
