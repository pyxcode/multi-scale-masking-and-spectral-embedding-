"""
Smoke test — forward pass complet avec données synthétiques.
Usage : python test_forward.py (depuis la racine du projet)

Fixes :
  - imgs : collated_batch déjà [B,1,H,W], pas de unsqueeze
  - gradient_pos_embed : tensor synthétique requis par GradTs_2dPE
  - add_w='origin' : obligatoire, False crash dans GradTs_2dPE
"""

import sys
import copy
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
sys.path.insert(0, './src')

from masks.spatialtemporal_multiblock import MaskCollator_fmri
from utils.tensors import repeat_interleave_batch
from masks.utils import apply_masks
from helper import init_model

# ── Config ────────────────────────────────────────────────────────────────────
B          = 2
HEIGHT     = 450
WIDTH      = 160
PATCH_SIZE = 16
DEVICE     = torch.device('cpu')

NUM_PATCHES          = HEIGHT * (WIDTH // PATCH_SIZE)  # 4500
gradient_pos_embed   = torch.zeros(NUM_PATCHES, 32)    # synthétique, shape [4500, 32]

print("="*55)
print("SMOKE TEST — FORWARD PASS")
print("="*55)

# ── 1. Collator ───────────────────────────────────────────────────────────────
print("\n[1] Collator...")
try:
    collator = MaskCollator_fmri(
        input_size=(HEIGHT, WIDTH),
        patch_size=PATCH_SIZE,
        multiscale_ts_scales=(0.03, 0.15, 0.45)
    )
    batch = [torch.randn(1, HEIGHT, WIDTH) for _ in range(B)]
    collated_batch, masks_enc_raw, masks_pred_raw = collator(batch)

    masks_enc  = [u.to(DEVICE) for u in masks_enc_raw]
    masks_pred = [u.to(DEVICE) for u in masks_pred_raw]
    imgs = collated_batch.to(DEVICE)  # déjà [B, 1, H, W]
    print(f"  ✅ imgs={imgs.shape}, masks_enc={masks_enc[0].shape}, masks_pred={masks_pred[0].shape}")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  ❌ CRASH collator : {e}")
    sys.exit(1)

# ── 2. Modèle ─────────────────────────────────────────────────────────────────
print("\n[2] Init modèle (vit_small)...")
try:
    encoder, predictor = init_model(
        device=DEVICE,
        patch_size=PATCH_SIZE,
        model_name='vit_small',
        crop_size=(HEIGHT, WIDTH),
        pred_depth=2,
        pred_emb_dim=192,
        gradient_pos_embed=gradient_pos_embed,
        attn_mode='normal',
        add_w='origin',
    )
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False
    print("  ✅ encoder + predictor + target_encoder OK")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  ❌ CRASH init_model : {e}")
    sys.exit(1)

# ── 3. Forward target ─────────────────────────────────────────────────────────
print("\n[3] Forward target...")
try:
    with torch.no_grad():
        h = target_encoder(imgs)
        h = F.layer_norm(h, (h.size(-1),))
        h = apply_masks(h, masks_pred)
        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
    print(f"  ✅ h={h.shape}")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  ❌ CRASH forward_target : {e}")
    sys.exit(1)

# ── 4. Forward context ────────────────────────────────────────────────────────
print("\n[4] Forward context...")
try:
    z = encoder(imgs, masks_enc, return_attention=False)
    z = predictor(z, masks_enc, masks_pred, return_attention=False)
    print(f"  ✅ z={z.shape}")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  ❌ CRASH forward_context : {e}")
    sys.exit(1)

# ── 5. Loss + backward ────────────────────────────────────────────────────────
print("\n[5] Loss + backward...")
try:
    loss = F.smooth_l1_loss(z, h)
    assert not torch.isnan(loss), "loss is nan"
    assert not torch.isinf(loss), "loss is inf"
    loss.backward()
    print(f"  ✅ loss={loss.item():.4f}, backward OK")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  ❌ CRASH loss/backward : {e}")
    sys.exit(1)

print("\n" + "="*55)
print("✅ FORWARD PASS VALIDÉ — prêt pour SIGReg")