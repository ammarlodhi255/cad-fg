# fig_comorbidities_bars.py
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Reproducibility of output paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "output_figures")
os.makedirs(out_dir, exist_ok=True)

# --- Color palette (expanded to avoid repeats) ---
# Use seven distinct colors so the last bars don't repeat earlier hues.
# Original five kept, plus two new distinct tones (lavender, teal).
PALETTE = [
    "#A5CDE2",  # light blue
    "#2078B3",  # deep blue
    "#B2DF8A",  # light green
    "#FEBE70",  # orange
    "#FA9B9A",  # salmon
    "#CAB2D6",  # lavender
    "#8DD3C7",  # teal
]

# --- Data (percent) ---
data = {
    "High blood pressure": 61.9,
    "Smoker": 45.2,
    "Diabetes mellitus": 40.5,
    "Dyslipidemia": 40.5,
    "Kidney failure": 14.3,
    "Heart failure": 14.3,
    "Atrial fibrillation": 4.8,
}

# Sort by value (descending)
items = sorted(data.items(), key=lambda x: x[1], reverse=True)
labels, values = zip(*items)

# --- Matplotlib global style ---
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 18,
})

fig, ax = plt.subplots(figsize=(8, 6))
y = np.arange(len(labels))

# Bars
# Use a unique color per bar (no wrap-around for this dataset size).
colors = PALETTE[: len(labels)]
bars = ax.barh(y, values, color=colors, edgecolor="#2b2b2b", linewidth=1.2)

# Labels and grid
ax.set_xlabel("Patients (%)")
ax.set_yticks(y, labels)
ax.invert_yaxis()  # highest at top
ax.grid(True, axis="x", which="major", linewidth=0.8, alpha=0.25)
ax.minorticks_on()
ax.grid(True, axis="x", which="minor", linewidth=0.4, alpha=0.15)
ax.set_axisbelow(True)

# Tight x-limits with headroom
max_v = max(values)
ax.set_xlim(0, max_v * 1.20)

# Annotate bars with values
for rect, val in zip(bars, values):
    # Always place labels outside, with extra right headroom.
    text_x = rect.get_width() + max_v * 0.01
    ax.text(text_x, rect.get_y() + rect.get_height() / 2, f"{val:.1f}%", va="center", ha="left", fontsize=13)

# Frame and margins
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)
fig.subplots_adjust(left=0.35, right=0.96, top=0.92, bottom=0.10)

ax.set_title("Comorbidities and Risk Factors")

# Save
fig.savefig(os.path.join(out_dir, "comorbidities_bars.pdf"))
fig.savefig(os.path.join(out_dir, "comorbidities_bars.svg"))
fig.savefig(os.path.join(out_dir, "comorbidities_bars.png"), dpi=200)
plt.show()
