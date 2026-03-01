"""
Test MaskCollator_fmri — basé sur la vraie structure du code.

Faits clés tirés du code :
  self.height = input_size[0]          = 450   (ROIs, pas divisé)
  self.width  = input_size[1] // patch  = 10   (160 // 16)
  TOTAL_PATCHES = 450 * 10             = 4500

  __call__ retourne :
    collated_batch          : tensor [B, 450, 160]
    collated_masks_enc      : list d'un tensor [B, N_enc]
    collated_masks_pred     : list d'un tensor [B, N_pred]
"""

import sys
import torch
sys.path.insert(0, '.')
from spatialtemporal_multiblock import MaskCollator_fmri

# ── Config ────────────────────────────────────────────────────────────────────
B           = 4
HEIGHT      = 450
WIDTH       = 160
PATCH_SIZE  = 16
# height interne = 450 (pas patchifié), width interne = 160//16 = 10
TOTAL_PATCHES = HEIGHT * (WIDTH // PATCH_SIZE)   # 4500

errors = []

def check(condition, ok_msg, fail_msg):
    if condition:
        print(f"  ✅ {ok_msg}")
    else:
        print(f"  ❌ {fail_msg}")
        errors.append(fail_msg)

# ── Appel ─────────────────────────────────────────────────────────────────────
batch = [torch.randn(HEIGHT, WIDTH) for _ in range(B)]
collator = MaskCollator_fmri(
    input_size=(HEIGHT, WIDTH),
    patch_size=PATCH_SIZE,
    multiscale_ts_scales=(0.03, 0.15, 0.45)
)

try:
    collated_batch, masks_enc_raw, masks_pred_raw = collator(batch)
except Exception as e:
    print(f"\n❌ CRASH au collator : {e}")
    sys.exit(1)

# ── 1. Types retournés ────────────────────────────────────────────────────────
print("\n[1] Types retournés")
print(f"    collated_batch  : {type(collated_batch)}")
print(f"    masks_enc_raw   : {type(masks_enc_raw)}")
print(f"    masks_pred_raw  : {type(masks_pred_raw)}")

# Le collator wrappe dans une liste via default_collate([[t], [t], ...])
# → masks_enc_raw est une list contenant 1 tensor [B, N]
check(isinstance(collated_batch, torch.Tensor),
      "collated_batch est un Tensor",
      f"collated_batch est {type(collated_batch)}, attendu Tensor")

# Unwrap proprement (sans masquer le bug, juste documenter)
if isinstance(masks_enc_raw, list):
    print(f"    → masks_enc est une liste de longueur {len(masks_enc_raw)}, unwrap [0]")
    masks_enc = masks_enc_raw[0]   # [B, N_enc]
else:
    masks_enc = masks_enc_raw

if isinstance(masks_pred_raw, list):
    print(f"    → masks_pred est une liste de longueur {len(masks_pred_raw)}, unwrap [0]")
    masks_pred = masks_pred_raw[0]  # [B, N_pred]
else:
    masks_pred = masks_pred_raw

# ── 2. Shapes ─────────────────────────────────────────────────────────────────
print("\n[2] Shapes")
print(f"    collated_batch : {collated_batch.shape}  (attendu [{B}, {HEIGHT}, {WIDTH}])")
print(f"    masks_enc      : {masks_enc.shape}        (attendu [{B}, N_enc])")
print(f"    masks_pred     : {masks_pred.shape}       (attendu [{B}, N_pred])")

check(collated_batch.shape == (B, HEIGHT, WIDTH),
      f"collated_batch shape OK {collated_batch.shape}",
      f"collated_batch shape {collated_batch.shape} ≠ ({B},{HEIGHT},{WIDTH})")

check(masks_enc.dim() == 2 and masks_enc.shape[0] == B,
      f"masks_enc shape OK {masks_enc.shape}",
      f"masks_enc shape incorrecte : {masks_enc.shape}")

check(masks_pred.dim() == 2 and masks_pred.shape[0] == B,
      f"masks_pred shape OK {masks_pred.shape}",
      f"masks_pred shape incorrecte : {masks_pred.shape}")

check(masks_pred.shape[1] > 0,
      f"masks_pred non vide ({masks_pred.shape[1]} patches)",
      f"masks_pred VIDE — 0 patches")

# ── 3. Indices dans les bounds ────────────────────────────────────────────────
print("\n[3] Indices dans [0, TOTAL_PATCHES)")
print(f"    TOTAL_PATCHES = {HEIGHT} * {WIDTH//PATCH_SIZE} = {TOTAL_PATCHES}")

if masks_enc.shape[1] > 0:
    enc_min, enc_max = masks_enc.min().item(), masks_enc.max().item()
    print(f"    masks_enc  : min={enc_min}, max={enc_max}")
    check(enc_min >= 0 and enc_max < TOTAL_PATCHES,
          f"masks_enc indices dans [0, {TOTAL_PATCHES})",
          f"masks_enc indices HORS BOUNDS : [{enc_min}, {enc_max}], max autorisé={TOTAL_PATCHES-1}")

if masks_pred.shape[1] > 0:
    pred_min, pred_max = masks_pred.min().item(), masks_pred.max().item()
    print(f"    masks_pred : min={pred_min}, max={pred_max}")
    check(pred_min >= 0 and pred_max < TOTAL_PATCHES,
          f"masks_pred indices dans [0, {TOTAL_PATCHES})",
          f"masks_pred indices HORS BOUNDS : [{pred_min}, {pred_max}], max autorisé={TOTAL_PATCHES-1}")

# ── 4. Sémantique enc > pred ──────────────────────────────────────────────────
print("\n[4] Contexte > Target")
if masks_enc.shape[1] > 0 and masks_pred.shape[1] > 0:
    check(masks_enc.shape[1] > masks_pred.shape[1],
          f"enc ({masks_enc.shape[1]}) > pred ({masks_pred.shape[1]})",
          f"enc ({masks_enc.shape[1]}) ≤ pred ({masks_pred.shape[1]}) — inversion probable")

# ── 5. Overlap et unicité par sujet ──────────────────────────────────────────
print("\n[5] Overlap & unicité (par sujet)")
if masks_enc.shape[1] > 0 and masks_pred.shape[1] > 0:
    for i in range(B):
        enc_list  = masks_enc[i].tolist()
        pred_list = masks_pred[i].tolist()
        enc_set   = set(enc_list)
        pred_set  = set(pred_list)
        overlap   = enc_set & pred_set
        dup_enc   = len(enc_list) - len(enc_set)
        dup_pred  = len(pred_list) - len(pred_set)
        coverage  = len(enc_set | pred_set) / TOTAL_PATCHES * 100
        print(f"    sujet {i}: enc={len(enc_set)}, pred={len(pred_set)}, "
              f"overlap={len(overlap)}, dup_enc={dup_enc}, dup_pred={dup_pred}, "
              f"coverage={coverage:.1f}%")
        if len(overlap) > 0:
            errors.append(f"sujet {i}: {len(overlap)} patches en overlap")

# ── 6. Multi-échelle — taille pred varie entre runs ──────────────────────────
print("\n[6] Multi-échelle (10 runs)")
pred_sizes = []
for _ in range(10):
    try:
        _, _, mp_raw = collator(batch)
        mp = mp_raw[0] if isinstance(mp_raw, list) else mp_raw
        pred_sizes.append(mp.shape[1])
    except Exception as e:
        pred_sizes.append(-1)

print(f"    tailles pred : {pred_sizes}")
unique = len(set(s for s in pred_sizes if s > 0))
check(unique >= 2,
      f"Variabilité multi-échelle OK ({unique} tailles différentes)",
      f"Pas de variabilité — toujours {pred_sizes[0]} patches (multiscale inactif ?)")

# ── Résumé ────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
if errors:
    print(f"❌ {len(errors)} ERREUR(S) :")
    for e in errors:
        print(f"   • {e}")
    sys.exit(1)
else:
    print("✅ TOUS LES TESTS PASSENT — masquage validé")
    sys.exit(0)