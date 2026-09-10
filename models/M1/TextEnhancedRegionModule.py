import torch
import torch.nn as nn
import torch.nn.functional as F

from models.M1.Alignment import TextToVisualAlignment

def hard_softmax(logits, dim):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret, index.squeeze(-2)

def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> torch.Tensor:
    # more stable https://github.com/pytorch/pytorch/issues/41663
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)
    origin_index = logits.max(dim, keepdim=True)[1]

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret, origin_index.squeeze(-2)

class TextEnhancedRegionModule(nn.Module):
    """
    使用预训练文本特征增强伪区域特征的模块
    集成Alignment中的对齐机制
    """

    def __init__(self, region_dim=512, text_dim=512, num_heads=8, dropout=0.1,
                 text_embeddings_path=None, text_categories=None, gumbel_tau=1.0,
                 assign_eps=1.0, gumbel=True, hard=True, sum_assign=False,
                 use_alignment=True, alpha_init=0.7):
        super().__init__()
        self.region_dim = region_dim
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.head_dim = region_dim // num_heads

        # 借鉴 MaskClipHead 的文本特征处理
        self.text_categories = text_categories
        self.text_embeddings_path = text_embeddings_path
        self.gumbel_tau = gumbel_tau
        self.assign_eps = assign_eps
        self.gumbel = gumbel
        self.hard = hard
        self.sum_assign = sum_assign
        self.use_alignment = use_alignment

        # 加载预训练文本特征
        if self.text_embeddings_path is not None:
            self.text_embeddings = nn.Parameter(torch.zeros(text_categories, text_dim))
            if self.text_embeddings_path.endswith('.pth') or self.text_embeddings_path.endswith('.pt'):
                self.load_text_embeddings()
            else:
                nn.init.normal_(self.text_embeddings, mean=0.0, std=0.01)
            self.text_embeddings.requires_grad = False
        else:
            self.text_embeddings = None

        # 区域特征投影层
        self.region_proj = nn.Sequential(
            nn.Linear(region_dim, region_dim),
            nn.LayerNorm(region_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 添加文本特征投影层（保持维度512，仅做特征变换）
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, text_dim),  # 输入输出维度相同（512→512）
            nn.LayerNorm(text_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 添加对齐模块
        if self.use_alignment:
            self.alignment_module = TextToVisualAlignment(
                dim=region_dim,
                alpha_init=alpha_init,
                use_projection=True,
                dropout=dropout
            )

        # 注意力机制
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=region_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 门控融合机制
        self.gate_proj = nn.Linear(region_dim * 2, region_dim)
        self.gate_sigmoid = nn.Sigmoid()

        # 输出层
        self.output_norm = nn.LayerNorm(region_dim)
        self.output_dropout = nn.Dropout(dropout)

    def load_text_embeddings(self):
        """从指定路径加载预训练文本嵌入"""
        if self.text_embeddings_path is not None:
            loaded = torch.load(self.text_embeddings_path, map_location='cuda')
            self.text_embeddings.data = loaded
            print(f'Loaded text embeddings from {self.text_embeddings_path}')

    def forward(self, region_feat, global_feat=None, return_attention=False):
        """
        前向传播
        Args:
            region_feat: 伪区域特征 [B, R, D_region]              [10, 128, 512]
            global_feat: 全局视觉特征 [B, D_region] (可选)         [10, 512]
            return_attention: 是否返回注意力权重
        Returns:
            enhanced_region: 增强后的区域特征 [B, R, D_region]
            attn_weights: 注意力权重 (可选)
        """
        batch_size, num_regions, _ = region_feat.shape

        # 如果没有提供全局特征，使用区域特征的均值
        if global_feat is None:
            global_feat = region_feat.mean(dim=1)  # [B, D_region]

        # 投影区域特征
        region_projected = self.region_proj(region_feat)  # [B, R, D_region]

        # 使用全局特征预选相关的文本特征（借鉴 MaskClipHead 的方法）
        if self.text_embeddings is not None:
            global_projected = global_feat

            # 计算全局特征与文本特征的相似度
            text_similarity = torch.matmul(
                global_projected,
                self.text_embeddings.t()  # [D_text, num_text]
            )  # [B, num_text]

            # 选择最相关的文本特征
            topk = min(15, self.text_categories)  # 选择前k个文本特征			# topk参数消融 5/10/15/20/25
            _, topk_indices = torch.topk(text_similarity, k=topk, dim=-1)  # [B, k]

            # 收集选中的文本特征
            selected_text_features = self.text_embeddings[topk_indices]  # [B, k, D_text]

            # 将文本特征投影到区域特征空间
            text_projected = self.text_proj(selected_text_features)  # [B, k, D_text]
        else:
            # 如果没有预训练文本特征，使用可学习的文本查询
            text_projected = self.text_queries.expand(batch_size, -1, -1)  # [B, num_queries, D_region]

        # 应用对齐机制：将区域特征映射到文本空间
        if self.use_alignment:
            region_aligned = self.alignment_module(
                text_features=text_projected,  # 文本作为对齐目标空间
                visual_features=region_projected  # 区域特征需要被对齐
            )
        else:
            region_aligned = region_projected

        # 文本作为查询，对齐后的区域作为键和值
        enhanced_region, attn_weights = self.cross_attn(
            query=text_projected,  # 文本作为查询 [B, k, D_region]
            key=region_aligned,  # 对齐后的区域作为键 [B, R, D_region]
            value=region_aligned,  # 对齐后的区域作为值 [B, R, D_region]
            need_weights=True
        )

        # 对注意力输出进行池化（从k个文本特征到R个区域特征）
        if enhanced_region.size(1) > 1:
            # 使用注意力权重加权平均
            enhanced_region = torch.bmm(
                attn_weights.transpose(1, 2),  # [B, R, k]
                enhanced_region  # [B, k, D_region]
            )  # [B, R, D_region]
        else:
            # 单个文本特征，直接扩展
            enhanced_region = enhanced_region.expand(-1, num_regions, -1)

        # 借鉴 MaskClipHead 的注意力处理方式
        if not self.sum_assign:
            enhanced_region = enhanced_region / (enhanced_region.sum(dim=-1, keepdim=True) + self.assign_eps)

        # 门控融合原始区域特征和文本增强特征
        combined = torch.cat([region_feat, enhanced_region], dim=-1)
        gate = self.gate_sigmoid(self.gate_proj(combined))
        enhanced_region = gate * region_feat + (1 - gate) * enhanced_region

        # 输出处理
        enhanced_region = self.output_norm(enhanced_region)
        enhanced_region = self.output_dropout(enhanced_region)

        if return_attention:
            return enhanced_region, attn_weights
        return enhanced_region

    # ... (保留原有的get_attn和calculate_similarity_loss方法)
    def get_attn(self, attn, gumbel=None, hard=None):
        """
        借鉴 MaskClipHead 的注意力计算方式
        """
        if gumbel is None:
            gumbel = self.gumbel
        if hard is None:
            hard = self.hard

        attn_dim = -2
        if gumbel and self.training:
            attn = gumbel_softmax(attn, dim=attn_dim, hard=hard, tau=self.gumbel_tau)
        else:
            if hard:
                attn = hard_softmax(attn, dim=attn_dim)
            else:
                attn = F.softmax(attn, dim=attn_dim)
        return attn

    def calculate_similarity_loss(self, region_features, text_features):
        """
        计算区域特征与文本特征的平均余弦相似度损失
        """
        # 归一化特征
        region_norm = F.normalize(region_features, p=2, dim=-1)  # [B, R, D]
        text_norm = F.normalize(text_features, p=2, dim=-1)  # [num_text, D]

        # 计算相似度矩阵 [B, R, num_text]
        similarity = torch.matmul(
            region_norm,
            text_norm.transpose(-2, -1)
        )

        # 获取每个区域最相似的文本特征
        max_similarity, _ = similarity.max(dim=-1)  # [B, R]

        # 计算平均最大相似度作为损失（取负值）
        loss = 1 - max_similarity.mean()

        return loss