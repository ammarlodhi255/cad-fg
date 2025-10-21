# fig_combined_stacked.py (fixed)
import os
from typing import List, Tuple

import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "output_figures")
os.makedirs(out_dir, exist_ok=True)

# ---- Helpers ---------------------------------------------------------------

def normalize_row(items: List[Tuple[str, float]]) -> list[float]:
    """Return widths that sum to exactly 100.0 (avoid rounding drift)."""
    vals = [v for _, v in items]
    s = sum(vals)
    if s == 0:
        return [0.0] * len(vals)
    widths = [v * 100.0 / s for v in vals]
    # force exact closure by adjusting the last slot
    widths[-1] = 100.0 - sum(widths[:-1])
    return widths

def fmt_pct(v: float) -> str:
    return f"{v:.1f}%"

# ---- Row-specific palettes (avoid cross-row semantic color reuse) ----------
# Involvement: Blues; LVEF: Green→Amber→Red; Indication: Purples
BLUES   = ["#A7CBE9", "#5AA2D1", "#1F78B4", "#0D4F7C"]
SEVERITY = ["#7BC96F", "#F4C95D", "#F39C34", "#E45756"]   # good→bad
PURPLES = ["#CAB2D6", "#9C89C9", "#6B5CA5", "#4B3D88"]

# ---- Data (percent) with clean, non-overlapping bins -----------------------

INVOLVEMENT = [
    ("<20%", 14.3),
    ("20–49%", 66.7),   # changed upper bound to avoid overlap with next bin
    ("50–69%", 0.0),    # explicit "gap" bucket (0%) so bins are exhaustive
    ("≥70%", 19.0),
]

LVEF = [
    ("Normal (≥55%)", 68.2),
    ("Mild (45–54%)", 9.8),
    ("Moderate (35–44%)", 0.0),  # non-overlapping with Mild
    ("Severe (<35%)", 22.0),
]

INDICATION = [
    ("Chronic coronary syndrome", 4.9),
    ("NSTE-ACS", 65.9),
    ("STE-ACS", 29.3),
]

# Assign colors per row (distinct palettes)
COLORS_INVOLVEMENT = [BLUES[0], BLUES[1], BLUES[2], BLUES[3]]
COLORS_LVEF        = SEVERITY                # already ordered good→bad
COLORS_INDICATION  = [PURPLES[0], PURPLES[2], PURPLES[3]]

ROWS: List[Tuple[str, List[Tuple[str, float]], List[str]]] = [
    ("Max involvement", INVOLVEMENT, COLORS_INVOLVEMENT),
    ("LVEF", LVEF, COLORS_LVEF),
    ("Indication", INDICATION, COLORS_INDICATION),
]

# ---- Matplotlib style ------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.titlesize": 18,
    }
)

fig, ax = plt.subplots(figsize=(11.5, 6.0))

y_positions = list(range(len(ROWS)))[::-1]
ax.set_xlim(0, 100)
ax.set_yticks(y_positions, [name for name, _, _ in ROWS])
ax.set_xlabel("Patients (%)")
ax.grid(True, axis="x", linewidth=0.8, alpha=0.25)
ax.minorticks_on()
ax.grid(True, axis="x", which="minor", linewidth=0.4, alpha=0.15)
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)

# ---- Draw rows -------------------------------------------------------------

for y, (row_name, items, colors) in zip(y_positions, ROWS):
    # widths used for geometry (normalized to 100); labels show original %
    widths = normalize_row(items)

    left = 0.0
    for (label, raw_value), width, color in zip(items, widths, colors):
        if raw_value > 0:
            # segment
            ax.barh(
                y,
                width,
                left=left,
                color=color,
                edgecolor="#2b2b2b",
                linewidth=0.9,
                height=0.62,
            )

            # annotation
            seg_start = left
            seg_end = left + width
            # show "label + %" when roomy; else just %.
            text_str = fmt_pct(raw_value)
            if width >= 25 and (seg_start >= 3) and (seg_end <= 97):
                text_str = f"{label}: {fmt_pct(raw_value)}"

            # position rules
            if seg_end > 97:
                x = min(99.0, seg_end - 1.0); ha = "right"
            elif seg_start < 3:
                x = max(1.0, seg_start + 1.0); ha = "left"
            else:
                x = seg_start + width / 2.0; ha = "center"

            ax.text(x, y, text_str, va="center", ha=ha, fontsize=12, clip_on=True)
            left += width
        else:
            # raw_value == 0 → don't draw a sliver; instead, add a clear callout
            ax.text(
                100.2, y,
                f"{label}: 0%",
                va="center", ha="left", fontsize=12, clip_on=False, color="#444444"
            )

# layout & title
fig.subplots_adjust(left=0.25, right=0.92, top=0.88, bottom=0.16)
fig.suptitle("Clinical Summary (separate 100% distributions with clean bins)")

# Export
fig.savefig(os.path.join(out_dir, "combined_stacked_fixed.pdf"))
fig.savefig(os.path.join(out_dir, "combined_stacked_fixed.svg"))
fig.savefig(os.path.join(out_dir, "combined_stacked_fixed.png"), dpi=200)

if __name__ == "__main__":
    plt.show()