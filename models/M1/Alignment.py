import torch
import torch.nn as nn
import torch.nn.functional as F


class TextToVisualAlignment(nn.Module):
    """
    单向对齐模块：只让视觉特征向文本特征空间靠拢
    """

    def __init__(self, dim, alpha_init=0.7, use_projection=True, dropout=0.1):
        super(TextToVisualAlignment, self).__init__()
        self.dim = dim
        self.use_projection = use_projection

        # 可学习的融合权重
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # 可选的投影层，增强表达能力
        if use_projection:
            self.proj = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(dim)
            )
        else:
            self.proj = nn.Identity()

    def forward(self, text_features, visual_features):
        """
        参数:
            text_features: 文本特征 [B, L, D] (作为对齐的目标空间)
            visual_features: 视觉特征 [B, M, D] (需要被对齐的特征)
        返回:
            对齐后的视觉特征 [B, M, D] (在文本空间中的视觉表示)
        """
        B, L, D = text_features.shape
        _, M, _ = visual_features.shape

        # 1. 计算文本特征的Gram矩阵 (x^T x)
        # Gram矩阵捕获了特征之间的相关性，代表了文本空间的几何结构
        gram_text = torch.matmul(text_features.transpose(1, 2), text_features)  # [B, D, D]

        # 2. 归一化Gram矩阵 (使用Frobenius范数)
        norm = torch.norm(gram_text, dim=(1, 2), keepdim=True) + 1e-8
        gram_text_normed = gram_text / norm

        # 3. 将视觉特征映射到文本空间
        # 这相当于将视觉特征投影到文本特征定义的空间中
        visual_mapped = torch.matmul(visual_features, gram_text_normed)  # [B, M, D]

        # 4. 增强映射特征的表达能力
        visual_mapped = self.proj(visual_mapped)

        # 5. 融合原始特征和映射特征
        # 使用可学习的alpha参数平衡原始视觉特征和文本空间映射后的特征
        visual_aligned = self.alpha * visual_features + (1 - self.alpha) * visual_mapped

        return visual_aligned

    def extra_repr(self):
        return f'dim={self.dim}, alpha={self.alpha.item():.3f}, use_projection={self.use_projection}'


class TextGuidedVisualEnhancer(nn.Module):
    """
    文本引导的视觉特征增强模块
    使用单向对齐将视觉特征映射到文本空间，增强其语义表示
    """

    def __init__(self, dim, num_heads=8, dropout=0.1, alpha_init=0.7):
        super(TextGuidedVisualEnhancer, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        # 单向对齐模块
        self.alignment = TextToVisualAlignment(
            dim, alpha_init, use_projection=True, dropout=dropout
        )

        # 交叉注意力机制（文本作为查询，视觉作为键值）
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 门控融合机制
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

        # 输出层
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_features, text_features):
        """
        参数:
            visual_features: 视觉特征 [B, M, D]
            text_features: 文本特征 [B, L, D] (作为引导信号)
        返回:
            增强后的视觉特征 [B, M, D]
        """
        # 1. 单向对齐：将视觉特征映射到文本空间
        visual_aligned = self.alignment(text_features, visual_features)

        # 2. 交叉注意力：使用文本特征进一步细化对齐后的视觉特征
        # 文本作为查询，对齐后的视觉特征作为键值
        visual_enhanced, _ = self.cross_attn(
            query=text_features,
            key=visual_aligned,
            value=visual_aligned
        )

        # 由于文本和视觉序列长度可能不同，我们需要将注意力输出适配回视觉序列长度
        if visual_enhanced.size(1) != visual_features.size(1):
            # 使用平均池化或插值来调整序列长度
            visual_enhanced = F.adaptive_avg_pool1d(
                visual_enhanced.transpose(1, 2),
                visual_features.size(1)
            ).transpose(1, 2)

        # 3. 门控融合：平衡原始视觉特征和增强后的特征
        gate = self.gate(torch.cat([visual_features, visual_enhanced], dim=-1))
        output = gate * visual_features + (1 - gate) * visual_enhanced

        # 4. 归一化和dropout
        output = self.norm(output)
        output = self.dropout(output)

        return output