# fig_indications_and_vessels.py
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, NullLocator

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "output_figures")
os.makedirs(out_dir, exist_ok=True)

# PALETTE = ["#A5CDE2", "#2078B3", "#B2DF8A", "#FEBE70", "#FA9B9A"]
PALETTE = ["#2078B3", "#FA9B9A", "#B2DF8A", "#FEBE70", "#FA9B9A"]

# Data
indications = {
    "Chronic coronary\nsyndrome": 4.9,
    "NSTE-ACS": 65.9,
    "STE-ACS": 29.3,
}
vessels = {"0": 23.8, "1": 50.0, "2": 14.3, "3": 11.9}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 18,
    }
)

# --- (A) Donut for clinical indications (separate figure) ---
fig1, ax1 = plt.subplots(figsize=(8, 5))
labels = list(indications.keys())
sizes = list(indications.values())
donut_colors = [PALETTE[1], PALETTE[3], PALETTE[4]]

wedges, _ = ax1.pie(
    sizes,
    labels=None,
    startangle=90,
    counterclock=False,
    colors=donut_colors,
    wedgeprops=dict(width=0.45, edgecolor="#2b2b2b", linewidth=1.0),
)

# Center circle label
ax1.text(0, 0, "Clinical\nIndication", ha="center", va="center", fontsize=14)

# Legend to the right; avoid line breaks from source labels
labels_clean = [lbl.replace("\n", " ") for lbl in labels]
legend_labels = [f"{lbl}: {val:.1f}%" for lbl, val in zip(labels_clean, sizes)]
# Place legend above the donut, centered across the figure
fig1.legend(
    wedges,
    legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.87),
    ncol=1,
    frameon=False,
)
ax1.set(aspect="equal")

fig1.suptitle("Angiography Indications")
fig1.subplots_adjust(left=0.08, right=0.98, top=0.74, bottom=0.12)

# Save figure 1
fig1.savefig(os.path.join(out_dir, "indications_donut.pdf"))
fig1.savefig(os.path.join(out_dir, "indications_donut.svg"))
fig1.savefig(os.path.join(out_dir, "indications_donut.png"), dpi=200)

# --- (B) Bars for number of vessels affected (separate figure) ---
fig2, ax2 = plt.subplots(figsize=(7.5, 5))
x = np.arange(len(vessels))
vals = list(vessels.values())

# Distinct colors per bar
bar_colors = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]]
bars = ax2.bar(x, vals, color=bar_colors, edgecolor="#2b2b2b", linewidth=1.2, width=0.65)

ax2.set_xticks(x, [f"{k} vessel{'s' if k!='1' else ''}" for k in vessels.keys()])
ax2.set_ylabel("Patients (%)")
ax2.set_title("Number of Vessels Affected")
ax2.set_ylim(0, max(vals) * 1.25)

# Grid and aesthetics (y only)
ax2.grid(True, axis="y", which="major", linewidth=0.8, alpha=0.25)
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.grid(True, axis="y", which="minor", linewidth=0.4, alpha=0.15)
ax2.xaxis.set_minor_locator(NullLocator())
ax2.set_axisbelow(True)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_linewidth(1.0)
ax2.spines["bottom"].set_linewidth(1.0)

# Annotate bars
max_v = max(vals)
for rect, v in zip(bars, vals):
    ax2.text(
        rect.get_x() + rect.get_width() / 2,
        rect.get_height() + max_v * 0.02,
        f"{v:.1f}%",
        ha="center",
        va="bottom",
        fontsize=13,
    )

fig2.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.18)

# Save figure 2
fig2.savefig(os.path.join(out_dir, "vessels_affected_bars.pdf"))
fig2.savefig(os.path.join(out_dir, "vessels_affected_bars.svg"))
fig2.savefig(os.path.join(out_dir, "vessels_affected_bars.png"), dpi=200)

plt.show()
