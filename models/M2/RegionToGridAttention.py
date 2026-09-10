# RegionToGridAttention.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class StandardCrossAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Scaling factor
        self.scale = self.head_dim ** -0.5

    def forward(self, grid_features, region_features):
        """
        Standard cross-attention from grid to region features

        Args:
            grid_features: [N, L_grid, D] grid features (query)
            region_features: [N, L_region, D] region features (key/value)
        Returns:
            Enhanced grid features [N, L_grid, D]
        """
        N = grid_features.shape[0]
        L_grid = grid_features.shape[1]
        L_region = region_features.shape[1]

        # Project queries, keys, values
        Q = self.q_proj(grid_features)  # [N, L_grid, D]
        K = self.k_proj(region_features)  # [N, L_region, D]
        V = self.v_proj(region_features)  # [N, L_region, D]

        # Reshape for multi-head attention
        Q = Q.view(N, L_grid, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [N, H, L_grid, d]
        K = K.view(N, L_region, self.num_heads, self.head_dim).permute(0, 2, 3, 1)  # [N, H, d, L_region]
        V = V.view(N, L_region, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [N, H, L_region, d]

        # Compute attention scores
        attn_scores = torch.matmul(Q, K) * self.scale  # [N, H, L_grid, L_region]
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [N, H, L_grid, d]

        # Combine heads and project
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(N, L_grid, self.embed_dim)
        attn_output = self.out_proj(attn_output)  # [N, L_grid, D]

        return attn_output


class DifferentialAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, depth=0):
        super().__init__()
        assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim // 2, embed_dim)

        # Differential attention parameters
        self.lambda_init = 0.8 - 0.6 * torch.exp(torch.tensor(-0.3 * depth))
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))

        # Sub-layer normalization
        self.subln = RMSNorm(embed_dim // 2, eps=1e-5)

        # Scaling factor
        self.scale = self.head_dim ** -0.5

    def forward(self, grid_features, region_features):
        """
        Differential attention from grid to region features

        Args:
            grid_features: [N, L_grid, D] grid features (query)
            region_features: [N, L_region, D] region features (key/value)
        Returns:
            Enhanced grid features [N, L_grid, D]
        """
        N = grid_features.shape[0]
        L_grid = grid_features.shape[1]
        L_region = region_features.shape[1]

        # Project queries, keys, values
        Q = self.q_proj(grid_features)  # [N, L_grid, D]
        K = self.k_proj(region_features)  # [N, L_region, D]
        V = self.v_proj(region_features)  # [N, L_region, D]

        # Reshape for multi-head attention
        Q = Q.view(N, L_grid, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [N, H, L_grid, d]
        K = K.view(N, L_region, self.num_heads, self.head_dim).permute(0, 2, 3, 1)  # [N, H, d, L_region]
        V = V.view(N, L_region, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [N, H, L_region, d]

        # Compute attention scores
        attn_scores = torch.matmul(Q, K) * self.scale  # [N, H, L_grid, L_region]

        # Apply differential mechanism
        lambda1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1)).clamp(max=5.0)
        lambda2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2)).clamp(max=5.0)
        lambda_full = lambda1 - lambda2 + self.lambda_init

        # Split attention heads into two groups
        attn1, attn2 = attn_scores.chunk(2, dim=1)
        attn_weights = attn1 - lambda_full * attn2  # [N, H/2, L_grid, L_region]
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to values
        V = V[:, :attn_weights.size(1)]  # Match head dimension
        attn_output = torch.matmul(attn_weights, V)  # [N, H/2, L_grid, d]

        # Combine heads and project
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(N, L_grid, self.embed_dim // 2)
        attn_output = self.subln(attn_output) * (1 - self.lambda_init)
        attn_output = self.out_proj(attn_output)  # [N, L_grid, D]

        return attn_output


class RegionToGridAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, depth=0, fusion_method='sum'):
        """
        Combined attention with standard and differential branches

        Args:
            embed_dim: Feature dimension (512)
            num_heads: Number of attention heads
            depth: Current layer depth (for differential attention)
            fusion_method: How to fuse the two attention outputs ('sum', 'concat', 'weighted')
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_method = fusion_method

        # Standard attention branch
        self.standard_attn = StandardCrossAttention(embed_dim, num_heads)

        # Differential attention branch
        self.diff_attn = DifferentialAttention(embed_dim, num_heads, depth)

        # Fusion parameters
        if fusion_method == 'weighted':
            self.alpha = nn.Parameter(torch.tensor(0.5))
        elif fusion_method == 'concat':
            self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim)

        # Output normalization
        self.norm = RMSNorm(embed_dim, eps=1e-5)

    def forward(self, grid_features, region_features):
        """
        Args:
            grid_features: [N, L_grid, D] grid features
            region_features: [N, L_region, D] region features
        Returns:
            Enhanced grid features [N, L_grid, D]
        """
        # Compute both attention outputs
        std_output = self.standard_attn(grid_features, region_features)
        diff_output = self.diff_attn(grid_features, region_features)

        # Fuse the outputs
        if self.fusion_method == 'sum':
            fused_output = std_output + diff_output
        elif self.fusion_method == 'concat':
            fused_output = self.fusion_proj(torch.cat([std_output, diff_output], dim=-1))
        elif self.fusion_method == 'weighted':
            fused_output = self.alpha * std_output + (1 - self.alpha) * diff_output

        # Normalize and add residual connection
        output = self.norm(fused_output)
        return grid_features + output