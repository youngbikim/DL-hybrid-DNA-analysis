#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INCSV = Path("stacking_analysis_out/summary_stacking_by_system_temperature.csv")
OUTDIR = Path("stacking_final_plots")
OUTDIR.mkdir(exist_ok=True)

df = pd.read_csv(INCSV)

system_order = ["all_D", "D9L9", "d6l6", "d3l3", "d2l2d2", "DL9"]

label_map = {
    "all_D": "all-D",
    "D9L9": "D9L9",
    "d6l6": "D6L6D6",
    "d3l3": "(D3L3)3",
    "d2l2d2": "(D2L2)4D2",
    "DL9": "(D1L1)9",
}

color_map = {
    "all_D": "black",
    "D9L9": "#D55E00",
    "d6l6": "#E69F00",
    "d3l3": "#0072B2",
    "d2l2d2": "#009E73",
    "DL9": "#CC79A7",
}

marker_map = {
    "all_D": "o",
    "D9L9": "s",
    "d6l6": "^",
    "d3l3": "v",
    "d2l2d2": "D",
    "DL9": "X",
}

systems = [s for s in system_order if s in df["system"].unique()]


def plot_metric(ax, metric, ylabel, title, ylim=None):
    for sys in systems:
        sub = df[df["system"] == sys].sort_values("temperature_K")
        if metric not in sub.columns:
            continue

        std_col = metric.replace("_mean", "_std")
        yerr = sub[std_col] if std_col in sub.columns else None

        ax.errorbar(
            sub["temperature_K"],
            sub[metric],
            yerr=yerr,
            marker=marker_map.get(sys, "o"),
            color=color_map.get(sys, None),
            linewidth=2.2,
            markersize=6.5,
            capsize=3,
            label=label_map.get(sys, sys),
        )

    ax.set_xlabel("Temperature (K)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=10)

    if ylim is not None:
        ax.set_ylim(*ylim)


# ============================================================
# Figure 1. Main 4-panel stacking figure
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

plot_metric(
    axes[0, 0],
    "stack_like_fraction_mean",
    "Stack-like fraction",
    "Overall stacking preservation",
    ylim=(0, 1.05),
)

plot_metric(
    axes[0, 1],
    "junction_stack_like_fraction_mean",
    "Junction stack-like fraction",
    "Stacking near D/L junctions",
    ylim=(0, 1.05),
)

plot_metric(
    axes[1, 0],
    "stack_contact_occupancy_mean",
    "Stacking contact occupancy",
    "Base-base contact persistence",
    ylim=(0, 1.05),
)

plot_metric(
    axes[1, 1],
    "stack_plane_angle_deg_mean",
    "Base plane angle (deg)",
    "Stacking geometry distortion",
    ylim=(0, 45),
)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.03),
    ncol=6,
    frameon=False,
    fontsize=10,
)

fig.suptitle(
    "Stacking cooperativity distinguishes long-block and high-density D/L junction duplexes",
    fontsize=15,
    weight="bold",
    y=1.08,
)

plt.tight_layout()
plt.savefig(OUTDIR / "main_stacking_metrics_4panel.png", dpi=300, bbox_inches="tight")
plt.savefig(OUTDIR / "main_stacking_metrics_4panel.pdf", bbox_inches="tight")
plt.close()


# ============================================================
# Figure 2. Focus on all-D / D9L9 / D6L6D6
# ============================================================
focus = ["all_D", "D9L9", "d6l6"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

for ax, metric, ylabel, title, ylim in [
    (
        axes[0],
        "stack_like_fraction_mean",
        "Stack-like fraction",
        "Overall stacking",
        (0, 1.05),
    ),
    (
        axes[1],
        "junction_stack_like_fraction_mean",
        "Junction stack-like fraction",
        "Junction-region stacking",
        (0, 1.05),
    ),
    (
        axes[2],
        "stack_plane_angle_deg_mean",
        "Base plane angle (deg)",
        "Stacking distortion",
        (0, 45),
    ),
]:
    for sys in focus:
        if sys not in df["system"].unique():
            continue

        sub = df[df["system"] == sys].sort_values("temperature_K")
        std_col = metric.replace("_mean", "_std")
        yerr = sub[std_col] if std_col in sub.columns else None

        ax.errorbar(
            sub["temperature_K"],
            sub[metric],
            yerr=yerr,
            marker=marker_map[sys],
            color=color_map[sys],
            linewidth=2.4,
            markersize=7,
            capsize=3,
            label=label_map[sys],
        )

    ax.set_xlabel("Temperature (K)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.08),
    ncol=3,
    frameon=False,
    fontsize=11,
)

fig.suptitle(
    "D9L9 and D6L6D6 retain cooperative stacking despite distorted geometry",
    fontsize=15,
    weight="bold",
    y=1.18,
)

plt.tight_layout()
plt.savefig(OUTDIR / "focus_allD_D9L9_D6L6_stacking.png", dpi=300, bbox_inches="tight")
plt.savefig(OUTDIR / "focus_allD_D9L9_D6L6_stacking.pdf", bbox_inches="tight")
plt.close()


# ============================================================
# Figure 3. Junction vs non-junction stacking
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))

for sys in systems:
    sub = df[df["system"] == sys].sort_values("temperature_K")

    if "junction_stack_like_fraction_mean" not in sub.columns:
        continue
    if "nonjunction_stack_like_fraction_mean" not in sub.columns:
        continue

    ax.plot(
        sub["temperature_K"],
        sub["junction_stack_like_fraction_mean"],
        marker=marker_map.get(sys, "o"),
        color=color_map.get(sys),
        linewidth=2.3,
        label=f"{label_map.get(sys, sys)} junction",
    )

    ax.plot(
        sub["temperature_K"],
        sub["nonjunction_stack_like_fraction_mean"],
        marker=marker_map.get(sys, "o"),
        color=color_map.get(sys),
        linewidth=1.6,
        linestyle="--",
        alpha=0.55,
    )

ax.set_xlabel("Temperature (K)", fontsize=12)
ax.set_ylabel("Stack-like fraction", fontsize=12)
ax.set_title(
    "Junction-local stacking is preserved only in long-block D/L duplexes\n"
    "solid = junction, dashed = non-junction",
    fontsize=13,
    weight="bold",
)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.25)
ax.legend(frameon=False, fontsize=8, ncol=2)

plt.tight_layout()
plt.savefig(OUTDIR / "junction_vs_nonjunction_stacking.png", dpi=300, bbox_inches="tight")
plt.savefig(OUTDIR / "junction_vs_nonjunction_stacking.pdf", bbox_inches="tight")
plt.close()


# ============================================================
# Figure 4. 300 K bar plot for mechanism summary
# ============================================================
target_T = 300
bar_df = df[df["temperature_K"] == target_T].copy()
bar_df["system"] = pd.Categorical(bar_df["system"], categories=system_order, ordered=True)
bar_df = bar_df.sort_values("system")

bar_df = bar_df[bar_df["system"].isin(systems)]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))

bar_metrics = [
    (
        "stack_like_fraction_mean",
        "Stack-like fraction",
        "Cooperative stacking",
        (0, 1.05),
    ),
    (
        "junction_stack_like_fraction_mean",
        "Junction stack-like fraction",
        "Junction stacking",
        (0, 1.05),
    ),
    (
        "stack_plane_angle_deg_mean",
        "Base plane angle (deg)",
        "Geometry distortion",
        (0, 45),
    ),
]

for ax, (metric, ylabel, title, ylim) in zip(axes, bar_metrics):
    xlabels = [label_map[s] for s in bar_df["system"]]
    colors = [color_map[s] for s in bar_df["system"]]

    ax.bar(
        range(len(bar_df)),
        bar_df[metric],
        color=colors,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.6,
    )

    std_col = metric.replace("_mean", "_std")
    if std_col in bar_df.columns:
        ax.errorbar(
            range(len(bar_df)),
            bar_df[metric],
            yerr=bar_df[std_col],
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1.0,
        )

    ax.set_xticks(range(len(bar_df)))
    ax.set_xticklabels(xlabels, rotation=35, ha="right")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.25)

fig.suptitle(
    "Stacking metrics at 300 K",
    fontsize=15,
    weight="bold",
)

plt.tight_layout()
plt.savefig(OUTDIR / "stacking_metrics_300K_bar.png", dpi=300, bbox_inches="tight")
plt.savefig(OUTDIR / "stacking_metrics_300K_bar.pdf", bbox_inches="tight")
plt.close()

print(f"[DONE] plots saved to {OUTDIR.resolve()}")
