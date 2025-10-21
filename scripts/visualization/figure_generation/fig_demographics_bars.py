# fig_demographics_bars.py
import os
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "output_figures")
os.makedirs(out_dir, exist_ok=True)

# Shared palette used across figures
PALETTE = ["#A5CDE2", "#2078B3", "#B2DF8A", "#FEBE70", "#FA9B9A", "#8DD3C7", "#CAB2D6"]

# Data
sex_labels = ["Female", "Male"]
sex_values = [47.6, 52.4]  # percentages

# Age summary (median and IQR/range)
age_text = "Age: 71.5 (58.25–78) years"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})

# Compact figure with room for a tight legend row
fig, ax = plt.subplots(figsize=(4.4, 3.0))

colors = [PALETTE[0], PALETTE[1]]
bars = ax.bar(sex_labels, sex_values, color=colors, edgecolor="#2b2b2b", linewidth=1.1, width=0.6)

# Axis formatting
ax.set_ylabel("Patients (%)")
ax.set_ylim(0, max(sex_values) * 1.20)
ax.grid(True, axis="y", which="major", linewidth=0.8, alpha=0.25)
ax.minorticks_on()
ax.grid(True, axis="y", which="minor", linewidth=0.4, alpha=0.15)
ax.set_axisbelow(True)

# Bar labels
for r, v in zip(bars, sex_values):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.6, f"{v:.1f}%", va="bottom", ha="center")

# Keep a consistent visible frame
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)

# Two-part legend to stay compact:
# 1) color-coded sex labels
leg1 = ax.legend(
    bars,
    sex_labels,
    loc="upper left",
    bbox_to_anchor=(0.02, 1.02),
    frameon=False,
    ncol=2,
    borderaxespad=0.0,
    columnspacing=0.8,
    handlelength=1.0,
    handletextpad=0.4,
)

# 2) age summary as text-only legend entry
age_handle = Line2D([], [], color="none", label=age_text)
leg2 = ax.legend(
    handles=[age_handle],
    loc="upper right",
    bbox_to_anchor=(0.98, 1.02),
    frameon=False,
    handlelength=0,
    handletextpad=0,
)
ax.add_artist(leg1)

ax.set_title("Demographics")

# Tight layout with compressed top margin to keep overall footprint small
fig.subplots_adjust(left=0.14, right=0.98, top=0.78, bottom=0.18)

# Save
fig.savefig(os.path.join(out_dir, "demographics_bars.pdf"))
fig.savefig(os.path.join(out_dir, "demographics_bars.svg"))
fig.savefig(os.path.join(out_dir, "demographics_bars.png"), dpi=200)

if __name__ == "__main__":
    plt.show()

