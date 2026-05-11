from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAttentionLayer(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            num_heads: int, 
            linear_dim: int, 
            num_tokens: int, 
            batch_size: int = 1,
            pairwise: bool = False
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        self.pairwise = pairwise
        self.batch_size = batch_size
        self.num_tokens = num_tokens

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Linformer projection matrices
        self.f_proj = nn.Linear(num_tokens, linear_dim, bias=False)
        self.e_proj = nn.Linear(num_tokens, linear_dim, bias=False)

        self.bias_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, num_heads)
        )

    def forward(self, x: torch.Tensor, pairwise_feats: Union[None, torch.Tensor] = None, key_padding_mask: Union[None, torch.Tensor] = None):
        B, N, E = self.batch_size, self.num_tokens, self.embed_dim

        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # B,H,N,head_dim
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # B,H,N,head_dim
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # B,H,N,head_dim
        
        # expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(-1).expand(B, 1, N, self.head_dim)  # B,1,N,head_dim
        expanded_mask = key_padding_mask.reshape(B, 1, N, 1).repeat(1, 1, 1, self.head_dim)  # B,1,N,head_dim
        K = K.masked_fill(expanded_mask, 0.0)
        V = V.masked_fill(expanded_mask, 0.0)

        K_prime = self.e_proj(K.transpose(2, 3)).transpose(2, 3) # B,H,linear_dim,head_dim
        V_prime = self.f_proj(V.transpose(2, 3)).transpose(2, 3) # B,H,linear_dim,head_dim
        
        scores = torch.matmul(Q, K_prime.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim))  # (B,H,N,head_dim)x(B,H,head_dim,linear_dim) => B,H,N,linear_dim
        attn = torch.softmax(scores, dim=-1)  # B,H,N,linear_dim
        out = torch.matmul(attn, V_prime)  # (B,H,N,linear_dim)x(B,H,linear_dim,head_dim) => B,H,N,head_dim

        out = out.transpose(1, 2).contiguous().view(B, N, E)
        out = self.out_proj(out)
        return out

class TransformerEncoderBlock(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            num_heads: int, 
            dim_feedforward: int = 2048, 
            dropout: float = 0.1, 
            linear_dim: Union[int, None] = None, 
            num_tokens: Union[int, None] = None,
            pairwise: bool = False,
            batch_size: int = 1,
        ):
        super().__init__()
        if linear_dim is not None and num_tokens is None:
            raise ValueError("num_tokens must be provided if linear_dim is specified")
        self.self_attn = LinearAttentionLayer(
            embed_dim, 
            num_heads, 
            linear_dim, 
            num_tokens, 
            batch_size=batch_size,
            pairwise=pairwise
        )
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(
            self, 
            src: torch.Tensor, 
            pairwise_feats: Union[None, torch.Tensor] = None, 
            src_key_padding_mask: Union[None, torch.Tensor] = None
        ):
        src2 = self.self_attn(src, pairwise_feats, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    """
    Transformer encoder dimension parameters:
    - num_features: input feature dimension
    - embed_size: dimension of the token embeddings and the CLS token
    - latent_dim: dimension of the output latent representation (after bottleneck)
    - num_heads: number of attention heads per layer
    - num_layers: number of transformer layers
    - linear_dim: if specified, use linear attention with this projection dimension
    - num_tokens: if using linear attention, the maximum number of tokens (including CLS) for projection
    - pairwise: whether to use pairwise bias in attention layers (requires pairwise_feats input); resource intensive!
    """
    def __init__(
            self, 
            num_features: int, 
            embed_size: int, 
            latent_dim: int, 
            num_heads: int = 8, 
            num_layers: int = 4,
            linear_dim: Union[int, None] = None,
            num_tokens: Union[int, None] = None,
            pairwise: bool = False,
            batch_size: int = 1
        ):
        super().__init__()
        self.input_proj = nn.Linear(num_features, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_size, 
                    num_heads, 
                    linear_dim=linear_dim, 
                    num_tokens=num_tokens+1 if num_tokens is not None else None,
                    pairwise=pairwise,
                    batch_size=batch_size
                ) for _ in range(num_layers)
            ]
        )
        self.norm_cls_embedding = nn.LayerNorm(embed_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.bottleneck = nn.Linear(embed_size, latent_dim)
        self.pairwise = pairwise # bool
        self.batch_size = batch_size
        self.num_tokens = num_tokens

    def forward(self, x: torch.Tensor, pairwise_feats: Union[None, torch.Tensor] = None, mask: Union[None, torch.Tensor] = None):
        B = self.batch_size
        x = self.input_proj(x) # [B, N, E]

        # cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = self.cls_token.repeat(B, 1, 1)  # (B, 1, embed_size)
        x = torch.cat([cls_tokens, x], dim=1) 
        device = x.device
            
        for layer in self.layers:
            x = layer(x, None, src_key_padding_mask=mask)

        cls_embedding = x[:, 0, :] # CLS token embedding
        latent = self.bottleneck(self.norm_cls_embedding(cls_embedding))
        return latent
    
class Projector(nn.Module):
    def __init__(self, input_dim, proj_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, z):
        z = self.net(z)
        norm = torch.sqrt(torch.sum(z**2, dim=-1, keepdim=True) + 1e-6)
        return z / norm
