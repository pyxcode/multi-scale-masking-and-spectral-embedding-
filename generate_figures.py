"""
Generate figures for Brain-JEPA paper (Clean B&W version)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse, Circle
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

BLACK = '#000000'
DARK_GRAY = '#505050'
GRAY = '#808080'
LIGHT_GRAY = '#D0D0D0'
WHITE = '#FFFFFF'


def figure_B_masking_strategy():
    """Figure B: Multi-Scale Temporal Masking Strategy - No overlaps"""
    fig, ax = plt.subplots(1, 1, figsize=(13, 5.5))
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 11)
    ax.axis('off')

    # Title
    ax.text(0, 10.5, 'A', fontsize=14, fontweight='bold')
    ax.text(5, 10.5, 'MULTI-SCALE TEMPORAL MASKING STRATEGY', fontsize=11, fontweight='bold')

    # Timeline - shifted right to make room for labels
    timeline_start = 45
    timeline_end = 175
    ax.plot([timeline_start, timeline_end], [9.2, 9.2], 'k-', linewidth=0.5)
    for i, label in enumerate(['0', '40 TR', '80 TR', '120 TR', '160 TR']):
        x = timeline_start + i * (timeline_end - timeline_start) / 4
        ax.plot([x, x], [9.1, 9.3], 'k-', linewidth=0.5)
        ax.text(x, 9.5, label, ha='center', fontsize=8)
    ax.text(180, 9.2, 'T = 160 TR (≈ 320 s)', fontsize=8, va='center')

    # Original Brain-JEPA - labels on separate line
    ax.text(0, 8, 'Original (Brain-JEPA)', fontsize=10, fontweight='bold')

    ctx = FancyBboxPatch((timeline_start, 7.2), 100, 0.7, boxstyle="round,pad=0.02",
                          facecolor=LIGHT_GRAY, edgecolor=BLACK, linewidth=1.5)
    ax.add_patch(ctx)
    ax.text(timeline_start + 50, 7.55, 'Context (encoder)', ha='center', va='center', fontsize=9)

    t1 = FancyBboxPatch((150, 7.2), 42, 0.7, boxstyle="round,pad=0.02",
                         facecolor=WHITE, edgecolor=BLACK, linewidth=1.5)
    ax.add_patch(t1)
    ax.text(171, 7.55, 'Target', ha='center', va='center', fontsize=9)

    ax.text(171, 6.4, '≈ 48 TR, single scale', ha='center', fontsize=8, color=GRAY, style='italic')

    # Ours - Multi-Scale - labels on separate line
    ax.text(0, 5.2, 'Ours (Multi-Scale)', fontsize=10, fontweight='bold')

    scales = [
        ('Short', '~10 s', 4.0, 15),
        ('Medium', '~48 s', 2.8, 35),
        ('Long', '~144 s', 1.6, 70),
    ]

    for label, time, y, target_width in scales:
        # Labels in left margin
        ax.text(0, y + 0.15, f'{label} ({time})', fontsize=9)

        # Context box
        ctx = FancyBboxPatch((timeline_start, y - 0.05), 90, 0.55, boxstyle="round,pad=0.02",
                              facecolor=LIGHT_GRAY, edgecolor=BLACK, linewidth=1)
        ax.add_patch(ctx)
        ax.text(timeline_start + 45, y + 0.2, 'Context', ha='center', va='center', fontsize=8)

        # Target box
        tgt = FancyBboxPatch((timeline_end - target_width, y - 0.05), target_width, 0.55,
                              boxstyle="round,pad=0.02",
                              facecolor=WHITE, edgecolor=BLACK, linewidth=1.5)
        ax.add_patch(tgt)

    # Right labels
    ax.text(180, 4.15, '↔ short', fontsize=8)
    ax.text(180, 2.95, '↔ medium', fontsize=8)
    ax.text(180, 1.75, '↔ long', fontsize=8)

    ax.text(110, 0.4, 'M_pred = ∪{short, medium, long}', ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig('figures/fig_A_masking_strategy.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: figures/fig_A_masking_strategy.png")


def figure_C_architecture():
    """Figure C: Model Architecture & Training Objective - Compact"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 4.5))
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 70)
    ax.axis('off')

    ax.text(0, 67, 'B', fontsize=14, fontweight='bold')
    ax.text(5, 67, 'MODEL ARCHITECTURE & TRAINING OBJECTIVE', fontsize=11, fontweight='bold')

    y_top = 48
    y_mid = 30
    y_bot = 12

    # === INPUT ===
    inp = FancyBboxPatch((2, y_mid - 7), 18, 14, boxstyle="round,pad=0.02",
                          facecolor=WHITE, edgecolor=GRAY, linewidth=1, linestyle='--')
    ax.add_patch(inp)
    ax.text(11, y_mid, 'fMRI\n[400,T]', ha='center', va='center', fontsize=8)

    ax.annotate('', xy=(24, y_mid), xytext=(20, y_mid),
                arrowprops=dict(arrowstyle='->', color=BLACK, lw=1))

    # === ONLINE ENCODER ===
    enc = FancyBboxPatch((26, y_mid - 8), 32, 16, boxstyle="round,pad=0.02",
                          facecolor=WHITE, edgecolor=BLACK, linewidth=1.5)
    ax.add_patch(enc)
    ax.text(42, y_mid + 2, 'Encoder', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(42, y_mid - 4, 'ViT-S', ha='center', va='center', fontsize=7, color=GRAY)

    # === TARGET ENCODER ===
    tgt_enc = FancyBboxPatch((26, y_top - 5), 32, 13, boxstyle="round,pad=0.02",
                              facecolor=LIGHT_GRAY, edgecolor=BLACK, linewidth=1)
    ax.add_patch(tgt_enc)
    ax.text(42, y_top + 2, 'Target Enc.', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(42, y_top - 3, '(EMA)', ha='center', va='center', fontsize=7, color=GRAY)

    ax.annotate('', xy=(64, y_mid), xytext=(58, y_mid),
                arrowprops=dict(arrowstyle='->', color=BLACK, lw=1))
    ax.text(61, y_mid + 3, 'z_ctx', ha='center', fontsize=7)

    # === PREDICTOR ===
    pred = FancyBboxPatch((66, y_mid - 8), 26, 16, boxstyle="round,pad=0.02",
                           facecolor=WHITE, edgecolor=BLACK, linewidth=1.5)
    ax.add_patch(pred)
    ax.text(79, y_mid + 2, 'Predictor', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(79, y_mid - 4, '6 blocks', ha='center', va='center', fontsize=7, color=GRAY)

    ax.annotate('', xy=(110, y_top), xytext=(58, y_top),
                arrowprops=dict(arrowstyle='->', color=BLACK, lw=1))
    ax.text(84, y_top + 3, 'z_tgt', ha='center', fontsize=7)

    ax.annotate('', xy=(110, y_mid), xytext=(92, y_mid),
                arrowprops=dict(arrowstyle='->', color=BLACK, lw=1))
    ax.text(101, y_mid + 3, 'ẑ_tgt', ha='center', fontsize=7)

    # === L_SMOOTH-L1 ===
    loss1 = FancyBboxPatch((112, y_top - 7), 38, 15, boxstyle="round,pad=0.02",
                            facecolor=WHITE, edgecolor=BLACK, linewidth=1)
    ax.add_patch(loss1)
    ax.text(131, y_top + 1, 'L_smooth-L1', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(131, y_top - 5, '‖z − ẑ‖₁', ha='center', va='center', fontsize=7)

    # === L_SIGREG ===
    loss2 = FancyBboxPatch((112, y_mid - 7), 38, 15, boxstyle="round,pad=0.02",
                            facecolor=LIGHT_GRAY, edgecolor=BLACK, linewidth=1)
    ax.add_patch(loss2)
    ax.text(131, y_mid + 1, 'L_SIGReg', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(131, y_mid - 5, 'λ‖Cov−I‖²', ha='center', va='center', fontsize=7)

    ax.annotate('', xy=(155, y_bot + 5), xytext=(150, y_top - 7),
                arrowprops=dict(arrowstyle='->', color=BLACK, lw=0.8))
    ax.annotate('', xy=(155, y_bot + 5), xytext=(150, y_mid - 7),
                arrowprops=dict(arrowstyle='->', color=BLACK, lw=0.8))

    # === TOTAL LOSS ===
    total = FancyBboxPatch((152, y_bot - 3), 40, 14, boxstyle="round,pad=0.02",
                            facecolor=DARK_GRAY, edgecolor=BLACK, linewidth=1.5)
    ax.add_patch(total)
    ax.text(172, y_bot + 4, 'L = L₁ + λL_S', ha='center', va='center',
            fontsize=9, fontweight='bold', color=WHITE)

    plt.tight_layout()
    plt.savefig('figures/fig_B_architecture.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: figures/fig_B_architecture.png")


def figure_D_results():
    """Figure D: Linear Probe Results - No overlaps"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    methods = ['Baseline', '+Multiscale', '+SIGReg', 'S+M']
    aucs = [0.542, 0.543, 0.556, 0.567]
    stds = [0.097, 0.094, 0.053, 0.068]
    colors = [WHITE, LIGHT_GRAY, GRAY, DARK_GRAY]

    x = np.arange(len(methods))
    bars = ax.bar(x, aucs, yerr=stds, capsize=4, color=colors, edgecolor=BLACK, linewidth=1.2,
                  error_kw={'linewidth': 1.2, 'capthick': 1.2}, width=0.65)

    for i, (v, s) in enumerate(zip(aucs, stds)):
        text_color = WHITE if i == 3 else BLACK
        ax.text(i, v - 0.015, f'{v:.3f}', ha='center', va='top', fontsize=11,
                fontweight='bold', color=text_color)
        ax.text(i, 0.44, f'±{s:.3f}', ha='center', va='top', fontsize=8, color=GRAY)

    ax.text(3, aucs[3] + stds[3] + 0.015, '★', ha='center', fontsize=14)

    ax.axhline(y=0.5, color=GRAY, linestyle='--', linewidth=1)
    ax.text(3.55, 0.502, 'chance', fontsize=8, color=GRAY, va='bottom')

    ax.set_ylabel('AUC-ROC', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(0.43, 0.68)
    ax.set_xlim(-0.5, 3.5)

    ax.set_title('C   LINEAR PROBE — GENDER CLASSIFICATION (AUC-ROC)',
                 loc='left', fontsize=11, fontweight='bold', pad=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('figures/fig_C_results.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: figures/fig_C_results.png")


if __name__ == '__main__':
    import os
    os.makedirs('figures', exist_ok=True)

    print("Generating figures...")
    figure_B_masking_strategy()
    figure_C_architecture()
    figure_D_results()
    print("\nDone! Figures in 'figures/'")
