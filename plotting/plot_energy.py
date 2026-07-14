#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = "."
OUTDIR = Path("ALL_INTERACTION_PLOTS")
OUTDIR.mkdir(exist_ok=True)

CSVNAME = "strand_interaction_energy_summary.csv"

# =========================
# plotting style
# =========================

system_order = [
    "all_D",
    "D9L9",
    "d6l6",
    "d3l3",
    "d2l2d2",
    "DL9",
]

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

# =========================
# helper functions
# =========================

def extract_temperature(path):
    parts = path.parts

    for p in reversed(parts):
        if re.fullmatch(r"\d{3}", p):
            val = int(p)
            if 200 <= val <= 400:
                return val

    return None

def extract_system(path):
    parts = path.parts

    for p in reversed(parts):
        if p in system_order:
            return p

    return None

# =========================
# collect all csv
# =========================

rows = []

for csvfile in Path(ROOT).rglob(CSVNAME):

    temp = extract_temperature(csvfile.parent)
    system = extract_system(csvfile.parent)

    if temp is None or system is None:
        continue

    try:
        df = pd.read_csv(csvfile)

        row = {
            "system": system,
            "temperature_K": temp,
            "Coul_SR_mean_kJmol": df["Coul_SR_mean_kJmol"].iloc[0],
            "LJ_SR_mean_kJmol": df["LJ_SR_mean_kJmol"].iloc[0],
            "total_mean_kJmol": df["total_mean_kJmol"].iloc[0],
            "total_mean_kcalmol": df["total_mean_kcalmol"].iloc[0],
            "Coul_SR_std_kJmol": df["Coul_SR_std_kJmol"].iloc[0],
            "LJ_SR_std_kJmol": df["LJ_SR_std_kJmol"].iloc[0],
            "total_std_kJmol": df["total_std_kJmol"].iloc[0],
        }

        rows.append(row)

    except Exception as e:
        print(f"[ERROR] {csvfile}: {e}")

all_df = pd.DataFrame(rows)

all_df = all_df.sort_values(["system", "temperature_K"])

all_df.to_csv(
    OUTDIR / "all_interaction_energy_summary.csv",
    index=False
)

print(all_df)

# =========================
# plotting helper
# =========================

def make_plot(ycol, yerr, ylabel, title, outfile):

    fig, ax = plt.subplots(figsize=(7,5))

    for sys in system_order:

        sub = all_df[all_df["system"] == sys]

        if len(sub) == 0:
            continue

        sub = sub.sort_values("temperature_K")

        ax.errorbar(
            sub["temperature_K"],
            sub[ycol],
            yerr=sub[yerr],
            marker=marker_map[sys],
            color=color_map[sys],
            linewidth=2.2,
            markersize=7,
            capsize=3,
            label=label_map[sys],
        )

    ax.set_xlabel("Temperature (K)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, weight="bold")

    ax.grid(True, alpha=0.25)

    # interaction energy는 더 negative가 안정
    ax.invert_yaxis()

    ax.legend(frameon=False, fontsize=10)

    plt.tight_layout()

    plt.savefig(
        OUTDIR / outfile,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =========================
# total interaction
# =========================

make_plot(
    ycol="total_mean_kcalmol",
    yerr="total_std_kJmol",
    ylabel="Inter-strand interaction energy (kcal/mol)",
    title="DNA strand interaction energy",
    outfile="total_interaction_energy.png",
)

# =========================
# Coulomb
# =========================

make_plot(
    ycol="Coul_SR_mean_kJmol",
    yerr="Coul_SR_std_kJmol",
    ylabel="Coul-SR interaction (kJ/mol)",
    title="Electrostatic inter-strand interaction",
    outfile="coulomb_interaction_energy.png",
)

# =========================
# LJ
# =========================

make_plot(
    ycol="LJ_SR_mean_kJmol",
    yerr="LJ_SR_std_kJmol",
    ylabel="LJ-SR interaction (kJ/mol)",
    title="van der Waals inter-strand interaction",
    outfile="lj_interaction_energy.png",
)

print(f"[DONE] plots saved to {OUTDIR.resolve()}")
