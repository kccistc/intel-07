import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    이미지를 고정된 크기의 패치로 나누고, 각 패치를 임베딩 벡터로 변환하는 클래스입니다.
    기존 CNN의 Conv 레이어와 유사한 역할을 수행하며, 트랜스포머 모델의 입력 형태에 맞춥니다.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 512):
        super().__init__()
        # 이미지 크기는 패치 크기로 나누어 떨어져야 합니다.
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid

        # Conv2d를 사용하여 패치 임베딩을 생성합니다.
        # kernel_size와 stride가 patch_size와 동일하여 겹치지 않는 패치를 만듭니다.
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 입력: [B, C, H, W]
        x = self.proj(x)
        # 출력: [B, embed_dim, grid, grid]
        # (flatten) -> [B, embed_dim, num_patches]
        # (transpose) -> [B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """
    단일 트랜스포머 인코더 블록을 정의합니다.
    Multi-Head Self-Attention과 MLP(Feed-Forward Network)로 구성됩니다.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.1,
    ):
        super().__init__()
        # 첫 번째 LayerNorm
        self.norm1 = nn.LayerNorm(dim)
        # Multi-Head Attention 레이어
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(proj_drop)

        # 두 번째 LayerNorm
        self.norm2 = nn.LayerNorm(dim)

        # MLP(다층 퍼셉트론) 레이어
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden, dim),
            nn.Dropout(proj_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LayerNorm -> Attention -> Dropout -> Residual Connection
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + self.drop1(attn_out)
        # LayerNorm -> MLP -> Residual Connection
        h = self.norm2(x)
        x = x + self.mlp(h)
        return x


class DetrDecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.1):
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.self_drop = nn.Dropout(proj_drop)

        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.cross_drop = nn.Dropout(proj_drop)

        self.ffn_norm = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(proj_drop),
            nn.Linear(hidden, dim), nn.Dropout(proj_drop),
        )

    def forward(self, tgt, memory, tgt_pos=None, mem_pos=None):
        # (a) self-attn on queries
        h = self.self_norm(tgt + (tgt_pos if tgt_pos is not None else 0))
        sa, _ = self.self_attn(h, h, h)
        tgt = tgt + self.self_drop(sa)

        # (b) cross-attn: queries attend to memory
        q = self.cross_norm(tgt + (tgt_pos if tgt_pos is not None else 0))
        k = memory + (mem_pos if mem_pos is not None else 0)
        ca, _ = self.cross_attn(q, k, k)
        tgt = tgt + self.cross_drop(ca)

        # (c) FFN
        h = self.ffn_norm(tgt)
        tgt = tgt + self.ffn(h)
        return tgt

class DetrDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, depth: int, proj_drop: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DetrDecoderLayer(embed_dim, num_heads, proj_drop=proj_drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, memory, mem_pos, query_embed):
        B = memory.size(0)
        Q, D = query_embed.size()
        # (수정) 쿼리 컨텐트는 0으로 초기화
        tgt = torch.zeros(B, Q, D, device=memory.device, dtype=memory.dtype)
        tgt_pos = query_embed.unsqueeze(0).repeat(B, 1, 1)  # 쿼리 '포지션'으로만 사용

        x = tgt
        for layer in self.layers:
            x = layer(x, memory, tgt_pos=tgt_pos, mem_pos=mem_pos)

        return self.norm(x)
    

class VisionTransformerDetection(nn.Module):
    """
    객체 감지 작업을 위한 Vision Transformer 모델입니다.
    이미지 분류용 ViT와 달리, 인코더-디코더 구조와 객체 예측 헤드를 가집니다.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 5,   # 옷의 종류 (T-Shirts, Jeans 등)
        num_queries: int = 100, # 동시에 예측할 수 있는 최대 객체 수
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        # 1. 인코더 부분: 이미지 특징 추출
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 분류 토큰 (여기서는 사용하지 않음)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))   # 위치 임베딩
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, proj_drop=drop_rate)
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # 2. 디코더 부분: 객체 쿼리를 기반으로 객체 예측
        self.query_embed = nn.Parameter(torch.randn(num_queries, embed_dim))
        self.decoder = DetrDecoder(embed_dim, num_heads, depth)

        # 3. 출력 헤드: 바운딩 박스 및 클래스 예측
        # 바운딩 박스를 예측하는 MLP 레이어 (x, y, w, h)
        self.bbox_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 4)  # x, y, 너비, 높이
        )
        # 클래스를 예측하는 선형 레이어 (+1은 '배경' 클래스)
        self.class_head = nn.Linear(embed_dim, num_classes + 1)

        self._init_weights()

    def _init_weights(self):
        # 모델 파라미터 초기화
        #nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.query_embed, std=0.02)
        nn.init.trunc_normal_(self.class_head.weight, std=0.02)
        if self.class_head.bias is not None:
            nn.init.zeros_(self.class_head.bias)

    def forward(self, x_list: list[torch.Tensor]):
        # 입력은 이미지 텐서들의 리스트입니다. 모든 텐서는 동일한 크기여야 합니다. (B, C, H, W)
        batch_size = len(x_list)
        assert batch_size > 0, "x_list must contain at least one image tensor"
        x = torch.stack(x_list, dim=0)  # [B, C, H, W]

        # 1) 인코더: 패치 임베딩 + 위치임베딩
        x = self.patch_embed(x)  # [B, N, D]
        # pos_embed 크기: [1, N, D]
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)

        for blk in self.encoder_blocks:
            x = blk(x)
        memory = self.encoder_norm(x)  # [B, N, D]

        # 2) 디코더: cross-attn (쿼리가 memory를 본다)
        mem_pos = self.pos_embed[:, :memory.size(1), :]  # [1, N, D]
        dec_out = self.decoder(memory, mem_pos, self.query_embed)  # [B, Q, D]

        # 3) 출력 헤드: bbox는 0~1 정규화 (cx, cy, w, h)
        pred_boxes = torch.sigmoid(self.bbox_head(dec_out))    # [B, Q, 4]
        pred_logits = self.class_head(dec_out)                 # [B, Q, num_classes+1]

        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}