# fig_lvef_bars.py
import os
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "output_figures")
os.makedirs(out_dir, exist_ok=True)

PALETTE = ["#A5CDE2", "#2078B3", "#B2DF8A", "#FEBE70", "#FA9B9A", "#8DD3C7", "#CAB2D6"]

# Data (percent)
data = {
    "Normal (>55%)": 68.2,
    "Mild dysfunction (45–55%)": 9.8,
    "Moderate dysfunction (35–45%)": 0.0,
    "Severe dysfunction (<35%)": 22.0,
}

labels = list(data.keys())
values = list(data.values())

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.titlesize": 18,
})

fig, ax = plt.subplots(figsize=(9, 5))
y = np.arange(len(labels))

colors = [PALETTE[2], PALETTE[3], PALETTE[0], PALETTE[4]]
bars = ax.barh(y, values, color=colors, edgecolor="#2b2b2b", linewidth=1.1)

ax.set_xlabel("Patients (%)")
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.grid(True, axis="x", linewidth=0.8, alpha=0.25)
ax.minorticks_on()
ax.grid(True, axis="x", which="minor", linewidth=0.4, alpha=0.15)
ax.set_axisbelow(True)

max_v = max(values) if values else 0
ax.set_xlim(0, max(100, max_v * 1.2))

for r, v in zip(bars, values):
    ax.text(r.get_width() + 0.5, r.get_y() + r.get_height()/2, f"{v:.1f}%", va="center")

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)

fig.subplots_adjust(left=0.40, right=0.96, top=0.88, bottom=0.15)
ax.set_title("Left Ventricular Ejection Fraction")

fig.savefig(os.path.join(out_dir, "lvef_bars.pdf"))
fig.savefig(os.path.join(out_dir, "lvef_bars.svg"))
fig.savefig(os.path.join(out_dir, "lvef_bars.png"), dpi=200)
plt.show()
