import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import Block

class MAE_ViT_Project(nn.Module):
    def __init__(self, vit_arch='vit_base_patch16_224', mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        
        # ViT Encoder Backbone
        # TIMM to create the standard ViT, but we will use its components manually
        self.vit = timm.create_model(vit_arch, pretrained=False, num_classes=0)
        
        # Extract metadata
        self.patch_size = self.vit.patch_embed.patch_size[0]
        self.embed_dim = self.vit.embed_dim
        self.num_patches = self.vit.patch_embed.num_patches
        
        # MAE Decoder
        self.decoder_embed_dim = 512
        self.decoder_depth = 8
        self.decoder_num_heads = 16
        
        # Project Encoder features -> Decoder dimension, Linear projection 768 -> 512
        self.decoder_embed = nn.Linear(self.embed_dim, self.decoder_embed_dim, bias=True)
        
        # Mask token (Learnable placeholder for missing pixels)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        
        # Positional Embedding for Decoder
        # (1 + num_patches) to account for potential class token
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.decoder_embed_dim), requires_grad=False
        )
        
        # Decoder Transformer Blocks
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=self.decoder_embed_dim,
                num_heads=self.decoder_num_heads,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm
            )
            for _ in range(self.decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(self.decoder_embed_dim)
        
        # Prediction Head: Decoder Dim -> Pixels per Patch
        # Output is (Batch, Patches, 16*16*3)
        self.decoder_pred = nn.Linear(
            self.decoder_embed_dim, 
            (self.patch_size**2) * 3, 
            bias=True
        )
        
        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        # Initialize position embeddings (sine-cosine approx or normal)
        torch.nn.init.normal_(self.decoder_pos_embed, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_vit_weights)

    def _init_vit_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # MASKING LOGIC (The Core of MAE)
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # Batch, Length, Dim
        len_keep = int(L * (1 - mask_ratio))
        
        # Generate noise for random masking
        noise = torch.rand(N, L, device=x.device)
        
        # argsort gives us the indices that would sort the array, for selecting random patches
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first 'len_keep' subsets
        ids_keep = ids_shuffle[:, :len_keep]
        
        # Gather the visible patches
        # We expand indices to [N, len_keep, D] to gather along the sequence dimension
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask in original order
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    # FORWARD PASS
    def forward_encoder(self, x):
        # 1. Embed Patches
        x = self.vit.patch_embed(x)
        
        # 2. Add Positional Embeddings (Standard ViT)
        # Note: timm's pos_embed includes class token, so we slice if needed
        # or we just add it. Here we assume simpler patch-only MAE for assignment clarity.
        pos_embed = self.vit.pos_embed[:, 1:, :] # Skip class token pos
        x = x + pos_embed
        
        # 3. Apply Masking
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        
        # 4. Apply Transformer Blocks (Encoder)
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)
        
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # 1. Embed tokens to decoder dimension
        x = self.decoder_embed(x)
        
        # 2. Append Mask Tokens to sequence
        # We have N_visible tokens. We need N_total tokens.
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        
        # We actually construct the full sequence by concatenating and shuffling back
        # Easier strategy: Concatenate (Visible + Masks) then Unshuffle
        # But for 'ids_restore', we need strict alignment.
        
        # Strategy: Create a placeholder for full sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no gradient on masks
        
        # 3. Unshuffle (Restore Order)
        # gather(input, dim, index)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        # 4. Add Decoder Positional Embeddings
        x = x_ + self.decoder_pos_embed[:, 1:, :] # Skip class token
        
        # 5. Apply Transformer Blocks (Decoder)
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # 6. Predict Pixels
        x = self.decoder_pred(x)
        return x

    def forward(self, imgs):
        # 1. Encode
        latent, mask, ids_restore = self.forward_encoder(imgs)
        
        # 2. Decode
        pred = self.forward_decoder(latent, ids_restore)
        
        # 3. Loss Calculation is usually external, but we return pred and mask
        return pred, mask

    def get_encoder(self):
        # Helper to extract backbone for Fine-Tuning
        return self.vit