# fig_coronary_involvement_donut.py
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "output_figures")
os.makedirs(out_dir, exist_ok=True)

# Shared palette
PALETTE = ["#A5CDE2", "#2078B3", "#B2DF8A", "#FEBE70", "#FA9B9A"]

# Data (percent)
data = {
    "< 20%": 14.3,
    "20%–50%": 66.7,
    "> 70%": 19.0,
}

labels = list(data.keys())
sizes = list(data.values())

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "axes.titlesize": 18,
    "figure.titlesize": 18,
})

# Colors convey severity: low=green, mid=orange, high=red
colors = [PALETTE[2], PALETTE[3], PALETTE[4]]

fig, ax = plt.subplots(figsize=(8, 5))
wedges, _ = ax.pie(
    sizes,
    labels=None,
    startangle=90,
    counterclock=False,
    colors=colors,
    # Thinner ring -> larger center hole so the label fits on one line
    wedgeprops=dict(width=0.28, edgecolor="#2b2b2b", linewidth=1.0),
)

ax.text(0, 0, "Max Involvement", ha="center", va="center", fontsize=16)
ax.set(aspect="equal")

legend_labels = [f"{l}: {v:.1f}%" for l, v in zip(labels, sizes)]
fig.legend(
    wedges,
    legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.90),
    ncol=1,
    frameon=False,
)

fig.suptitle("Maximum Degree of Coronary Artery Involvement")
fig.subplots_adjust(left=0.08, right=0.98, top=0.64, bottom=0.12)

fig.savefig(os.path.join(out_dir, "coronary_involvement_donut.pdf"))
fig.savefig(os.path.join(out_dir, "coronary_involvement_donut.svg"))
fig.savefig(os.path.join(out_dir, "coronary_involvement_donut.png"), dpi=200)
plt.show()
