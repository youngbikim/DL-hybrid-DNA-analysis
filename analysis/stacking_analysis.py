#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

# ============================================================
# settings
# ============================================================

ROOT = "."
TOPO = "em.gro"
TRAJ = "pro.xtc"

OUTDIR = Path("stacking_analysis_out")
OUTDIR.mkdir(exist_ok=True)

BP_PER_STRAND = 18
TOTAL_RES = 36

USE_LAST_FRACTION = 0.5
FRAME_STRIDE = 10

STACK_DIST_CUTOFF = 4.5      # adjacent base COM distance cutoff
STACK_CONTACT_CUTOFF = 4.5   # heavy atom contact cutoff
PLANE_ANGLE_CUTOFF = 35.0    # degrees; smaller = more parallel/stack-like

BACKBONE_NAMES = {
    "P", "OP1", "OP2", "O1P", "O2P",
    "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "O2'",
    "O5*", "C5*", "C4*", "O4*", "C3*", "O3*", "C2*", "C1*", "O2*"
}

SYSTEM_JUNCTION_BP = {
    "alld":  [],
    "d9l9":  [8, 9, 10],
    "d6l6":  [5, 6, 11, 12],
    "d3l3":  [2, 3, 5, 6, 8, 9, 11, 12],
    "d2l2":  [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    "dl9":   list(range(18)),
    "d1l1":  list(range(18)),
}

JUNCTION_NEIGHBOR = 1

# ============================================================
# helper
# ============================================================

def extract_temperature(folder):
    folder = Path(folder)
    name = folder.name

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
    root = Path(ROOT).resolve()
    parent = system_dir.parent.resolve()

    if parent != root:
        return system_dir.parent.name
    return system_dir.name


def normalize_system_key(system_name):
    return (
        system_name.lower()
        .replace("_", "")
        .replace("-", "")
        .replace("(", "")
        .replace(")", "")
        .replace(" ", "")
    )


def get_junction_bp_indices(system_name):
    normalized = normalize_system_key(system_name)

    for key in sorted(SYSTEM_JUNCTION_BP.keys(), key=len, reverse=True):
        if key in normalized:
            return SYSTEM_JUNCTION_BP[key]

    return []


def get_junction_stack_indices(system_name, bp_per_strand=18):
    """
    adjacent stacking index는 bp i - bp i+1 사이를 의미.
    junction bp 주변 ±1을 stacking pair index로 변환.
    """
    junction_bp = get_junction_bp_indices(system_name)

    if len(junction_bp) == 0:
        return []

    stack_idx = set()

    for ji in junction_bp:
        for delta in range(-JUNCTION_NEIGHBOR, JUNCTION_NEIGHBOR + 1):
            bp_i = ji + delta

            # stacking pair는 i-i+1, 범위 0~16
            if 0 <= bp_i < bp_per_strand - 1:
                stack_idx.add(bp_i)

            if 0 <= bp_i - 1 < bp_per_strand - 1:
                stack_idx.add(bp_i - 1)

    return sorted(stack_idx)


def find_system_dirs(root):
    dirs = []

    for gro in Path(root).rglob(TOPO):
        xtc = gro.parent / TRAJ
        if xtc.exists():
            dirs.append(gro.parent)

    return sorted(set(dirs))


def get_nucleic_residues(u):
    nuc = u.select_atoms("nucleic")

    if len(nuc.residues) >= TOTAL_RES:
        return list(nuc.residues)[:TOTAL_RES]

    return list(u.residues)[:TOTAL_RES]


def get_base_atoms(res):
    idx = []

    for a in res.atoms:
        nm = a.name.upper()

        if nm in BACKBONE_NAMES:
            continue
        if nm.startswith("H"):
            continue

        idx.append(a.ix)

    if len(idx) == 0:
        return res.atoms[:0]

    return res.universe.atoms[np.array(idx, dtype=int)]


def base_com(res):
    ag = get_base_atoms(res)
    if len(ag) == 0:
        return None
    return ag.center_of_geometry()


def base_plane_normal(res):
    """
    base heavy atoms에 PCA를 적용해서 plane normal vector 계산.
    base가 평면에 가까울수록 가장 작은 singular vector가 normal.
    """
    ag = get_base_atoms(res)

    if len(ag) < 3:
        return None

    coords = ag.positions.astype(float)
    coords -= coords.mean(axis=0)

    try:
        _, _, vh = np.linalg.svd(coords, full_matrices=False)
        normal = vh[-1]
        norm = np.linalg.norm(normal)

        if norm < 1e-8:
            return None

        return normal / norm

    except Exception:
        return None


def angle_between_normals(n1, n2):
    if n1 is None or n2 is None:
        return np.nan

    cosang = np.dot(n1, n2)
    cosang = np.clip(cosang, -1.0, 1.0)

    # plane normal은 방향성이 ±라서 abs 사용
    angle = np.degrees(np.arccos(abs(cosang)))

    return float(angle)


def stacking_contact(res1, res2):
    ag1 = get_base_atoms(res1)
    ag2 = get_base_atoms(res2)

    if len(ag1) == 0 or len(ag2) == 0:
        return np.nan, np.nan

    dmat = distance_array(ag1.positions, ag2.positions)
    min_dist = float(np.min(dmat))
    n_contacts = int(np.sum(dmat < STACK_CONTACT_CUTOFF))

    contact_occ = 1.0 if min_dist < STACK_CONTACT_CUTOFF else 0.0

    return contact_occ, n_contacts


def stacking_pair_metrics(res_a, res_b):
    c1 = base_com(res_a)
    c2 = base_com(res_b)

    if c1 is None or c2 is None:
        com_dist = np.nan
    else:
        com_dist = float(np.linalg.norm(c1 - c2))

    n1 = base_plane_normal(res_a)
    n2 = base_plane_normal(res_b)
    angle = angle_between_normals(n1, n2)

    contact_occ, n_contacts = stacking_contact(res_a, res_b)

    if np.isnan(com_dist) or np.isnan(angle) or np.isnan(contact_occ):
        stack_like = np.nan
    else:
        stack_like = (
            1.0
            if (com_dist < STACK_DIST_CUTOFF and angle < PLANE_ANGLE_CUTOFF and contact_occ > 0.5)
            else 0.0
        )

    return {
        "com_distance_A": com_dist,
        "plane_angle_deg": angle,
        "contact_occupancy": contact_occ,
        "n_base_contacts": n_contacts,
        "stack_like": stack_like,
    }


def safe_mean(vals):
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if len(vals) else np.nan


# ============================================================
# core analysis
# ============================================================

def analyze_system(system_dir):
    system_dir = Path(system_dir)

    print(f"[INFO] analyzing {system_dir}")

    u = mda.Universe(str(system_dir / TOPO), str(system_dir / TRAJ))

    system = extract_system_name(system_dir)
    temp = extract_temperature(system_dir)

    residues = get_nucleic_residues(u)

    if len(residues) < TOTAL_RES:
        raise ValueError(f"Need {TOTAL_RES} nucleic residues, found {len(residues)}")

    residues = residues[:TOTAL_RES]

    strand1 = residues[:BP_PER_STRAND]
    strand2 = residues[BP_PER_STRAND:TOTAL_RES]

    # strand2는 reverse complement 방향이므로 bp order와 맞추려면 뒤집음
    strand2_bp_order = list(reversed(strand2))

    junction_stack_idx = get_junction_stack_indices(system, BP_PER_STRAND)

    n_frames = len(u.trajectory)
    start = int((1.0 - USE_LAST_FRACTION) * n_frames)
    use_indices = list(range(start, n_frames, FRAME_STRIDE))

    if len(use_indices) == 0:
        raise ValueError("No frames selected")

    frame_rows = []
    pair_rows = []

    for fi in use_indices:
        ts = u.trajectory[fi]

        per_pair = []

        # adjacent base stacking은 각 strand 내부에서 i-i+1
        # bp index 기준으로 strand1[i]-strand1[i+1],
        # strand2_bp_order[i]-strand2_bp_order[i+1]를 같이 평가.
        for i in range(BP_PER_STRAND - 1):
            r1a = strand1[i]
            r1b = strand1[i + 1]

            r2a = strand2_bp_order[i]
            r2b = strand2_bp_order[i + 1]

            m1 = stacking_pair_metrics(r1a, r1b)
            m2 = stacking_pair_metrics(r2a, r2b)

            row = {
                "system": system,
                "folder": str(system_dir),
                "temperature_K": temp,
                "time_ps": ts.time,
                "stack_index": i,
                "is_junction_near": i in junction_stack_idx,

                "strand1_com_distance_A": m1["com_distance_A"],
                "strand1_plane_angle_deg": m1["plane_angle_deg"],
                "strand1_contact_occupancy": m1["contact_occupancy"],
                "strand1_n_base_contacts": m1["n_base_contacts"],
                "strand1_stack_like": m1["stack_like"],

                "strand2_com_distance_A": m2["com_distance_A"],
                "strand2_plane_angle_deg": m2["plane_angle_deg"],
                "strand2_contact_occupancy": m2["contact_occupancy"],
                "strand2_n_base_contacts": m2["n_base_contacts"],
                "strand2_stack_like": m2["stack_like"],
            }

            row["mean_com_distance_A"] = safe_mean([
                row["strand1_com_distance_A"],
                row["strand2_com_distance_A"],
            ])
            row["mean_plane_angle_deg"] = safe_mean([
                row["strand1_plane_angle_deg"],
                row["strand2_plane_angle_deg"],
            ])
            row["mean_contact_occupancy"] = safe_mean([
                row["strand1_contact_occupancy"],
                row["strand2_contact_occupancy"],
            ])
            row["mean_n_base_contacts"] = safe_mean([
                row["strand1_n_base_contacts"],
                row["strand2_n_base_contacts"],
            ])
            row["mean_stack_like"] = safe_mean([
                row["strand1_stack_like"],
                row["strand2_stack_like"],
            ])

            per_pair.append(row)
            pair_rows.append(row)

        all_pairs = per_pair
        junc_pairs = [x for x in per_pair if x["is_junction_near"]]
        nonjunc_pairs = [x for x in per_pair if not x["is_junction_near"]]

        frame_rows.append({
            "system": system,
            "folder": str(system_dir),
            "temperature_K": temp,
            "time_ps": ts.time,

            "stack_com_distance_A_mean": safe_mean([x["mean_com_distance_A"] for x in all_pairs]),
            "stack_plane_angle_deg_mean": safe_mean([x["mean_plane_angle_deg"] for x in all_pairs]),
            "stack_contact_occupancy_mean": safe_mean([x["mean_contact_occupancy"] for x in all_pairs]),
            "stack_n_base_contacts_mean": safe_mean([x["mean_n_base_contacts"] for x in all_pairs]),
            "stack_like_fraction": safe_mean([x["mean_stack_like"] for x in all_pairs]),

            "junction_stack_com_distance_A_mean": safe_mean([x["mean_com_distance_A"] for x in junc_pairs]),
            "junction_stack_plane_angle_deg_mean": safe_mean([x["mean_plane_angle_deg"] for x in junc_pairs]),
            "junction_stack_contact_occupancy_mean": safe_mean([x["mean_contact_occupancy"] for x in junc_pairs]),
            "junction_stack_n_base_contacts_mean": safe_mean([x["mean_n_base_contacts"] for x in junc_pairs]),
            "junction_stack_like_fraction": safe_mean([x["mean_stack_like"] for x in junc_pairs]),

            "nonjunction_stack_com_distance_A_mean": safe_mean([x["mean_com_distance_A"] for x in nonjunc_pairs]),
            "nonjunction_stack_plane_angle_deg_mean": safe_mean([x["mean_plane_angle_deg"] for x in nonjunc_pairs]),
            "nonjunction_stack_contact_occupancy_mean": safe_mean([x["mean_contact_occupancy"] for x in nonjunc_pairs]),
            "nonjunction_stack_n_base_contacts_mean": safe_mean([x["mean_n_base_contacts"] for x in nonjunc_pairs]),
            "nonjunction_stack_like_fraction": safe_mean([x["mean_stack_like"] for x in nonjunc_pairs]),
        })

    return pd.DataFrame(frame_rows), pd.DataFrame(pair_rows)


def main():
    system_dirs = find_system_dirs(ROOT)

    print("[INFO] Found directories:")
    for sd in system_dirs:
        print("   ", sd)

    all_frames = []
    all_pairs = []

    for sd in system_dirs:
        try:
            frames_df, pair_df = analyze_system(sd)
            all_frames.append(frames_df)
            all_pairs.append(pair_df)
        except Exception as e:
            print(f"[ERROR] {sd}: {e}")

    if len(all_frames) == 0:
        raise RuntimeError("No systems analyzed")

    frames_all = pd.concat(all_frames, ignore_index=True)
    pairs_all = pd.concat(all_pairs, ignore_index=True)

    frames_all.to_csv(OUTDIR / "all_frames_stacking.csv", index=False)
    pairs_all.to_csv(OUTDIR / "all_pairwise_stacking.csv", index=False)

    summary = frames_all.groupby(["system", "temperature_K"]).agg(
        stack_com_distance_A_mean=("stack_com_distance_A_mean", "mean"),
        stack_com_distance_A_std=("stack_com_distance_A_mean", "std"),

        stack_plane_angle_deg_mean=("stack_plane_angle_deg_mean", "mean"),
        stack_plane_angle_deg_std=("stack_plane_angle_deg_mean", "std"),

        stack_contact_occupancy_mean=("stack_contact_occupancy_mean", "mean"),
        stack_contact_occupancy_std=("stack_contact_occupancy_mean", "std"),

        stack_n_base_contacts_mean=("stack_n_base_contacts_mean", "mean"),
        stack_n_base_contacts_std=("stack_n_base_contacts_mean", "std"),

        stack_like_fraction_mean=("stack_like_fraction", "mean"),
        stack_like_fraction_std=("stack_like_fraction", "std"),

        junction_stack_com_distance_A_mean=("junction_stack_com_distance_A_mean", "mean"),
        junction_stack_com_distance_A_std=("junction_stack_com_distance_A_mean", "std"),

        junction_stack_plane_angle_deg_mean=("junction_stack_plane_angle_deg_mean", "mean"),
        junction_stack_plane_angle_deg_std=("junction_stack_plane_angle_deg_mean", "std"),

        junction_stack_contact_occupancy_mean=("junction_stack_contact_occupancy_mean", "mean"),
        junction_stack_contact_occupancy_std=("junction_stack_contact_occupancy_mean", "std"),

        junction_stack_n_base_contacts_mean=("junction_stack_n_base_contacts_mean", "mean"),
        junction_stack_n_base_contacts_std=("junction_stack_n_base_contacts_mean", "std"),

        junction_stack_like_fraction_mean=("junction_stack_like_fraction", "mean"),
        junction_stack_like_fraction_std=("junction_stack_like_fraction", "std"),

        nonjunction_stack_like_fraction_mean=("nonjunction_stack_like_fraction", "mean"),
        nonjunction_stack_like_fraction_std=("nonjunction_stack_like_fraction", "std"),

        n_frames=("time_ps", "count"),
    ).reset_index()

    summary.to_csv(OUTDIR / "summary_stacking_by_system_temperature.csv", index=False)

    print(f"[DONE] saved to {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
