# RegionToGridFusionModule.py
import torch
import torch.nn as nn
import torch.nn.functional as F
# Import the new attention module
from models.M2.RegionToGridAttention import RegionToGridAttention

class RegionToGridFusionModule(nn.Module):
    def __init__(self, dim, num_regions=64, base_topk=8, min_topk=2, max_topk=16,
                 gumbel=False, hard=False, assign_eps=1e-6, use_learnable_region=True,
                 use_attention=False, attn_heads=8, attn_fusion='sum'):  # Updated parameter names
        super().__init__()
        self.dim = dim
        self.num_regions = num_regions
        self.base_topk = base_topk
        self.min_topk = min_topk
        self.max_topk = max_topk
        self.gumbel = gumbel
        self.hard = hard
        self.assign_eps = assign_eps
        self.use_learnable_region = use_learnable_region
        self.use_attention = use_attention  # Updated parameter name

        # Adaptive TopK scaling factor
        self.scale_factor = 0.5

        # Attention module
        if use_attention:
            self.attention = RegionToGridAttention(
                embed_dim=dim,
                num_heads=attn_heads,  # Updated parameter name
                depth=0,
                fusion_method=attn_fusion
            )

    def forward(self, grid_feat, region_feat):
        B, N, C = grid_feat.shape

        # Dynamic TopK calculation
        topk = self.calculate_adaptive_topk(grid_feat)

        # Generate global token using max pooling
        global_token, _ = torch.max(grid_feat, dim=1, keepdim=True)

        # Use learnable region or input region
        if region_feat is None:
            region_feat = self.region_token.expand(B, -1, -1)  # [B, R, C]
        B, R, C = region_feat.shape

        # Global token and region feature matching
        g_score = torch.einsum('brc,bkc->br', region_feat, global_token)

        # Use dynamic TopK to select regions
        topk_val, topk_idx = torch.topk(g_score, k=min(topk, R), dim=1)
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, C)
        selected_region = torch.gather(region_feat, dim=1, index=gather_idx)  # [B, k, C]

        # Use attention for feature alignment
        if self.use_attention:
            aligned_feat = self.attention(grid_feat, selected_region)
            return aligned_feat
        else:
            # Original dot product attention method
            sim = torch.matmul(selected_region, grid_feat.transpose(1, 2))
            sim = self.tau * sim
            attn = self.get_attn(sim)
            assign_attn = attn / (attn.sum(dim=1, keepdim=True) + self.assign_eps)
            aligned_feat = torch.bmm(assign_attn.transpose(1, 2), selected_region)
            return aligned_feat

    def calculate_adaptive_topk(self, grid_feat):
        """
        Calculate adaptive TopK value based on feature complexity
        """
        B, N, C = grid_feat.shape

        # Calculate feature complexity (feature variance)
        feat_var = torch.var(grid_feat, dim=(1, 2)).mean()  # [B] -> scalar

        # Calculate dynamic TopK value
        dynamic_topk = self.base_topk * (1 + self.scale_factor * feat_var)

        # Ensure within [min_topk, max_topk] range
        topk = torch.clamp(dynamic_topk, min=self.min_topk, max=self.max_topk).int()

        return topk

    def get_attn(self, attn_logits):
        if self.gumbel and self.training:
            return F.gumbel_softmax(attn_logits, tau=self.tau, hard=self.hard, dim=1)
        elif self.hard:
            hard_idx = attn_logits.argmax(dim=1, keepdim=True)
            return torch.zeros_like(attn_logits).scatter_(1, hard_idx, 1.0)
        else:
            return F.softmax(attn_logits, dim=1)