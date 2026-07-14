#!/usr/bingg/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

ROOT = "."
TOPO = "em.gro"
TRAJ = "pro.xtc"
OUTDIR = Path("terminal_binding_proxy_analysis")
OUTDIR.mkdir(exist_ok=True)

BP_PER_STRAND = 18
TOTAL_RES = 36
FRAME_STRIDE = 10
USE_LAST_FRACTION = 0.5

HBOND_CUTOFF = 3.5
CONTACT_CUTOFF = 4.5

BASEPAIR_HEAVY_ATOM_RULES = {
    ("A", "T"): [("N6", "O4"), ("N1", "N3")],
    ("T", "A"): [("O4", "N6"), ("N3", "N1")],
    ("G", "C"): [("O6", "N4"), ("N1", "N3"), ("N2", "O2")],
    ("C", "G"): [("N4", "O6"), ("N3", "N1"), ("O2", "N2")],
}

BACKBONE_NAMES = {
    "P", "OP1", "OP2", "O1P", "O2P",
    "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "O2'",
    "O5*", "C5*", "C4*", "O4*", "C3*", "O3*", "C2*", "C1*", "O2*"
}

def norm_base(resname):
    rn = resname.upper()
    if "DA" in rn: return "A"
    if "DT" in rn: return "T"
    if "DG" in rn: return "G"
    if "DC" in rn: return "C"
    return rn

def get_atom_pos(residue, atom_name):
    ag = residue.atoms.select_atoms(f"name {atom_name}")
    if len(ag) == 0:
        return None
    return ag.positions[0]

def get_nucleic_residues(u):
    nuc = u.select_atoms("nucleic")
    if len(nuc.residues) >= TOTAL_RES:
        return list(nuc.residues)[:TOTAL_RES]
    return list(u.residues)[:TOTAL_RES]

def get_base_heavy_atoms(res):
    idx = []
    for a in res.atoms:
        if a.name.upper() in BACKBONE_NAMES:
            continue
        if a.name.upper().startswith("H"):
            continue
        idx.append(a.ix)
    return idx

def residues_to_atomgroup(u, residues):
    idx = np.concatenate([r.atoms.indices for r in residues])
    return u.atoms[idx]

def bp_pairs():
    return [(i, 2 * BP_PER_STRAND - 1 - i) for i in range(BP_PER_STRAND)]

def bp_formed(res1, res2):
    b1 = norm_base(res1.resname)
    b2 = norm_base(res2.resname)
    key = (b1, b2)

    if key not in BASEPAIR_HEAVY_ATOM_RULES:
        return np.nan

    for a1, a2 in BASEPAIR_HEAVY_ATOM_RULES[key]:
        p1 = get_atom_pos(res1, a1)
        p2 = get_atom_pos(res2, a2)
        if p1 is None or p2 is None:
            return np.nan
        if np.linalg.norm(p1 - p2) > HBOND_CUTOFF:
            return 0.0
    return 1.0

def bp_opening_distance(res1, res2):
    idx1 = get_base_heavy_atoms(res1)
    idx2 = get_base_heavy_atoms(res2)
    if len(idx1) == 0 or len(idx2) == 0:
        return np.nan
    c1 = res1.universe.atoms[idx1].center_of_geometry()
    c2 = res2.universe.atoms[idx2].center_of_geometry()
    return float(np.linalg.norm(c1 - c2))

def extract_temperature(folder):
    name = Path(folder).name
    if re.fullmatch(r"\d{2,4}", name):
        val = int(name)
        if 200 <= val <= 500:
            return val
    m = re.search(r"(?<!\d)(\d{3})(?!\d)", name)
    if m:
        val = int(m.group(1))
        if 200 <= val <= 500:
            return val
    return np.nan

def extract_system_name(system_dir):
    system_dir = Path(system_dir)
    parent = system_dir.parent
    if parent.resolve() != Path(ROOT).resolve():
        return parent.name
    return system_dir.name

def find_system_dirs(root):
    dirs = []
    for gro in Path(root).rglob(TOPO):
        xtc = gro.parent / TRAJ
        if xtc.exists():
            dirs.append(gro.parent)
    return sorted(set(dirs))

def analyze_one(system_dir):
    system_dir = Path(system_dir)
    u = mda.Universe(str(system_dir / TOPO), str(system_dir / TRAJ))

    system = extract_system_name(system_dir)
    temp = extract_temperature(system_dir)

    residues = get_nucleic_residues(u)
    if len(residues) < TOTAL_RES:
        raise ValueError(f"Need {TOTAL_RES} residues, found {len(residues)}")

    strand1 = residues[:BP_PER_STRAND]
    strand2 = residues[BP_PER_STRAND:TOTAL_RES]

    strand1_ag = residues_to_atomgroup(u, strand1).select_atoms("not name H*")
    strand2_ag = residues_to_atomgroup(u, strand2).select_atoms("not name H*")

    n_frames = len(u.trajectory)
    start = int((1.0 - USE_LAST_FRACTION) * n_frames)
    use_indices = list(range(start, n_frames, FRAME_STRIDE))

    rows = []

    for fi in use_indices:
        ts = u.trajectory[fi]

        per_bp = {}
        opening = {}

        for i, j in bp_pairs():
            formed = bp_formed(residues[i], residues[j])
            dist = bp_opening_distance(residues[i], residues[j])
            per_bp[i] = formed
            opening[i] = dist

        terminal_indices = [0, 1, 16, 17]
        end_indices = [0, 17]
        near_end_indices = [1, 16]
        internal_indices = list(range(1, 17))
        core_indices = list(range(2, 16))

        def mean_bp(indices):
            vals = [per_bp[i] for i in indices if not np.isnan(per_bp[i])]
            return float(np.mean(vals)) if vals else np.nan

        def mean_open(indices):
            vals = [opening[i] for i in indices if not np.isnan(opening[i])]
            return float(np.mean(vals)) if vals else np.nan

        # strand-strand contact proxy
        dmat = distance_array(strand1_ag.positions, strand2_ag.positions)
        contact_count = int(np.sum(dmat < CONTACT_CUTOFF))
        contact_density = contact_count / (len(strand1_ag) * len(strand2_ag))

        # inverse-distance interaction proxy
        # 값이 클수록 strand 간 packing/contact가 강하다고 해석
        safe_d = dmat[dmat > 1.0]
        inv_distance_score = float(np.mean(1.0 / safe_d)) if len(safe_d) else np.nan

        rows.append({
            "system": system,
            "folder": str(system_dir),
            "temperature_K": temp,
            "time_ps": ts.time,

            "terminal_bp_fraction": mean_bp(terminal_indices),
            "end_bp_fraction": mean_bp(end_indices),
            "near_end_bp_fraction": mean_bp(near_end_indices),
            "internal_bp_fraction": mean_bp(internal_indices),
            "core_bp_fraction": mean_bp(core_indices),

            "terminal_opening_A": mean_open(terminal_indices),
            "end_opening_A": mean_open(end_indices),
            "near_end_opening_A": mean_open(near_end_indices),
            "internal_opening_A": mean_open(internal_indices),
            "core_opening_A": mean_open(core_indices),

            "strand_contact_count": contact_count,
            "strand_contact_density": contact_density,
            "strand_inverse_distance_score": inv_distance_score,
        })

    return pd.DataFrame(rows)

def main():
    all_rows = []

    system_dirs = find_system_dirs(ROOT)
    print("[INFO] Found directories:")
    for sd in system_dirs:
        print(" ", sd)

    for sd in system_dirs:
        try:
            print(f"[INFO] analyzing {sd}")
            df = analyze_one(sd)
            all_rows.append(df)
        except Exception as e:
            print(f"[ERROR] {sd}: {e}")

    if not all_rows:
        raise RuntimeError("No systems analyzed.")

    frames = pd.concat(all_rows, ignore_index=True)
    frames.to_csv(OUTDIR / "all_frames_terminal_binding_proxy.csv", index=False)

    summary = frames.groupby(["system", "temperature_K"]).agg(
        terminal_bp_fraction_mean=("terminal_bp_fraction", "mean"),
        terminal_bp_fraction_std=("terminal_bp_fraction", "std"),
        end_bp_fraction_mean=("end_bp_fraction", "mean"),
        near_end_bp_fraction_mean=("near_end_bp_fraction", "mean"),
        internal_bp_fraction_mean=("internal_bp_fraction", "mean"),
        core_bp_fraction_mean=("core_bp_fraction", "mean"),

        terminal_opening_A_mean=("terminal_opening_A", "mean"),
        end_opening_A_mean=("end_opening_A", "mean"),
        near_end_opening_A_mean=("near_end_opening_A", "mean"),
        internal_opening_A_mean=("internal_opening_A", "mean"),
        core_opening_A_mean=("core_opening_A", "mean"),

        strand_contact_count_mean=("strand_contact_count", "mean"),
        strand_contact_density_mean=("strand_contact_density", "mean"),
        strand_inverse_distance_score_mean=("strand_inverse_distance_score", "mean"),
        n_frames=("time_ps", "count"),
    ).reset_index()

    summary.to_csv(OUTDIR / "summary_terminal_binding_proxy.csv", index=False)

    print(f"[DONE] saved to {OUTDIR.resolve()}")

if __name__ == "__main__":
    main()
