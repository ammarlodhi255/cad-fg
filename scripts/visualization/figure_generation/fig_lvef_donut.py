# fig_lvef_donut.py
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "output_figures")
os.makedirs(out_dir, exist_ok=True)

# Shared palette across figures
PALETTE = ["#A5CDE2", "#2078B3", "#B2DF8A", "#FEBE70", "#FA9B9A"]

# Data (percent)
data = {
    "Normal (>55%)": 68.2,
    "Mild dysfunction (45–55%)": 9.8,
    "Moderate dysfunction (35–45%)": 0.0,
    "Severe dysfunction (<35%)": 22.0,
}

labels = list(data.keys())
sizes = list(data.values())

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "axes.titlesize": 18,
    "figure.titlesize": 18,
})

# Donut colors (match severity semantics: green, orange, blue, red)
colors = [PALETTE[2], PALETTE[3], PALETTE[0], PALETTE[4]]

fig, ax = plt.subplots(figsize=(8, 5))
wedges, _ = ax.pie(
    sizes,
    labels=None,
    startangle=90,
    counterclock=False,
    colors=colors,
    wedgeprops=dict(width=0.45, edgecolor="#2b2b2b", linewidth=1.0),
)

# Center label
ax.text(0, 0, "LVEF", ha="center", va="center", fontsize=16)
ax.set(aspect="equal")

# Legend above with percentages
legend_labels = [f"{l}: {v:.1f}%" for l, v in zip(labels, sizes)]
fig.legend(
    wedges,
    legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.90),
    ncol=1,
    frameon=False,
)

fig.suptitle("Left Ventricular Ejection Fraction")
fig.subplots_adjust(left=0.08, right=0.98, top=0.62, bottom=0.12)

# Save
fig.savefig(os.path.join(out_dir, "lvef_donut.pdf"))
fig.savefig(os.path.join(out_dir, "lvef_donut.svg"))
fig.savefig(os.path.join(out_dir, "lvef_donut.png"), dpi=200)
plt.show()
