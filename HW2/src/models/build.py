# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer


def build_model(num_classes):
    model = SwinTransformer(
        img_size=384,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[ 4, 8, 16, 32 ],
        window_size=12,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        use_checkpoint=False
    )

    return model
