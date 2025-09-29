import torch
import torch.nn as nn
import torchvision.models as models

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
        # (a) self-attn
        h = self.self_norm(tgt + (tgt_pos if tgt_pos is not None else 0))
        sa, _ = self.self_attn(h, h, h)
        tgt = tgt + self.self_drop(sa)

        # (b) cross-attn
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
        tgt = torch.zeros(B, Q, D, device=memory.device, dtype=memory.dtype)
        tgt_pos = query_embed.unsqueeze(0).repeat(B, 1, 1)

        x = tgt
        for layer in self.layers:
            x = layer(x, memory, tgt_pos=tgt_pos, mem_pos=mem_pos)
        return self.norm(x)

class VisionTransformerDetection(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        num_queries: int = 100,
        decoder_depth: int = 6,
        decoder_heads: int = 8,
    ):
        super().__init__()
        # 1. Pretrained ViT backbone (ImageNet 사전학습)
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        
        # 원본 ViT의 위치 임베딩을 저장하고, 백본에서는 사용하지 않도록 None으로 설정합니다.
        self.pos_embed = vit.encoder.pos_embedding
        vit.encoder.pos_embedding = None
        
        self.backbone_layers = vit.encoder.layers  # 개별 레이어에 접근
        self.patch_embed = vit.conv_proj
        self.encoder_norm = vit.encoder.ln  # 마지막 레이어 정규화

        self.embed_dim = vit.hidden_dim

        # 2. 디코더 (DETR 구조)
        self.query_embed = nn.Parameter(torch.randn(num_queries, self.embed_dim))
        self.decoder = DetrDecoder(self.embed_dim, decoder_heads, decoder_depth)

        # 3. 출력 헤드
        self.bbox_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, 4)
        )
        self.class_head = nn.Linear(self.embed_dim, num_classes + 1)

    def forward(self, x_list: list[torch.Tensor]):
        B = len(x_list)
        x = torch.stack(x_list, dim=0)

        # (a) 패치 임베딩
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # (b) 위치 임베딩을 수동으로 더하고, 트랜스포머 인코더 레이어를 순차적으로 통과
        # [CLS] 토큰 위치 임베딩을 제외하고 이미지 패치 임베딩만 사용합니다.
        pos_embed = self.pos_embed[:, 1:1+x.size(1), :]
        x = x + pos_embed
        
        for layer in self.backbone_layers:
            x = layer(x)
        
        memory = self.encoder_norm(x)

        # (c) 디코더
        dec_out = self.decoder(memory, pos_embed, self.query_embed)

        # (d) 예측
        pred_boxes = torch.sigmoid(self.bbox_head(dec_out))
        pred_logits = self.class_head(dec_out)

        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}