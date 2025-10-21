# fig_lvef_stacked.py
import os
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "output_figures")
os.makedirs(out_dir, exist_ok=True)

PALETTE = ["#A5CDE2", "#2078B3", "#B2DF8A", "#FEBE70", "#FA9B9A"]

# Data (percent)
lvef = {
    "Normal (>55%)": 68.2,
    "Mild (45–55%)": 9.8,
    "Moderate (35–45%)": 0.0,
    "Severe (<35%)": 22.0
}
labels = list(lvef.keys())
values = list(lvef.values())

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

fig, ax = plt.subplots(figsize=(8, 5))

# Single stacked bar at x=0
x0 = 0
left = 0
colors = [PALETTE[2], PALETTE[3], PALETTE[0], PALETTE[4]]
for lab, val, col in zip(labels, values, colors):
    ax.bar(x0, val, bottom=left, color=col, edgecolor="#2b2b2b", linewidth=1.1, width=0.6, label=lab)
    if val > 1e-6:
        ax.text(x0, left + val/2, f"{val:.1f}%", ha="center", va="center", fontsize=13)
    left += val

# Cosmetics
ax.set_xlim(-0.9, 0.9)
ax.set_ylim(0, 100)
ax.set_xticks([x0], ["LVEF"])
ax.set_ylabel("Patients (%)")
ax.grid(True, axis="y", linewidth=0.8, alpha=0.25)
ax.minorticks_on()
ax.grid(True, axis="y", which="minor", linewidth=0.4, alpha=0.15)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)

# Legend on top
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)

ax.set_title("Left Ventricular Ejection Fraction")

fig.subplots_adjust(left=0.12, right=0.96, top=0.80, bottom=0.15)
fig.savefig(os.path.join(out_dir, "lvef_stacked.pdf"))
fig.savefig(os.path.join(out_dir, "lvef_stacked.svg"))
fig.savefig(os.path.join(out_dir, "lvef_stacked.png"), dpi=200)
plt.show()
