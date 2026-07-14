#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DNA D/L complex nuclease proxy analysis
----------------------------------------
주요 수정 사항:
  1. [BUG FIX] sum(AtomGroup generator, start=...) 패턴 -> mda.Merge / concatenate 방식으로 교체
     → 'list' object has no attribute 'atoms' 에러 해결
  2. [NEW] --t 옵션: 분석할 최대 시뮬레이션 시간 지정 (예: --t=50ns, --t=100ps)
  3. [NEW] --last-fraction 옵션: 뒤에서 몇 %를 쓸지 (기본 50%)
  4. [SAFE] trajectory가 짧아도 frame이 1개 이상이면 분석 시도
"""

import re
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

# ============================================================
# SETTINGS (CLI로 override 가능한 것들은 main()에서 처리)
# ============================================================
ROOT_DIR = "."
TOPO_NAME = "em.gro"
TRAJ_NAME = "pro.xtc"
LOG_NAME = "pro.log"

OUTDIR = Path("analysis_out")

BP_PER_STRAND = 18
TOTAL_RES = BP_PER_STRAND * 2

USE_LAST_FRACTION = 0.5   # --last-fraction 으로 override
FRAME_STRIDE = 10
MAX_TIME_PS = None        # --t 옵션으로 설정됨 (None = 제한 없음)

HBOND_DIST_CUTOFF = 3.5
NATIVE_CONTACT_CUTOFF = 4.5
WATER_CONTACT_CUTOFF = 3.5
BASE_SHIELD_CUTOFF = 5.0
OPP_STRAND_CONTACT_CUTOFF = 5.0
GROOVE_PP_CUTOFF = 18.0

JUNCTION_RESINDICES = [7, 8, 9, 10, 25, 26, 27, 28]

# ============================================================
# D/L junction 위치 정의 (시퀀스 기반, strand1 0-indexed)
# underline 위치에서 D→L 또는 L→D 전환이 일어나는 bp index
# bp_pairs[i] = (i, 35-i) 이므로 strand1 index = bp index
# ============================================================
# 시퀀스: 5'-CGAACATTCCTAAGTCTG-3' (18mer, index 0~17)
# strand2는 bp_per_strand 이후 residue들로 자동 매핑됨
SYSTEM_JUNCTION_BP = {
    # key: 소문자+특수문자 제거 후 부분 매칭. 구체적인(긴) 패턴 우선.
    # value: junction bp index 목록 (strand1 기준, 0-indexed)
    "alld":  [],                                     # all_D: junction 없음
    "d9l9":  [8, 9, 10],                             # D9L9: 9번 전환 ±1
    "d6l6":  [5, 6, 11, 12],                         # D6L6D6: 6번, 12번 전환
    "d3l3":  [2, 3, 5, 6, 8, 9, 11, 12],             # (D3L3)3: 3개 단위 전환점
    "d2l2":  [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],  # (D2L2)4D2: 2개 단위
    "dl9":   list(range(18)),                         # (D1L1)9: 매 1개마다
    "d1l1":  list(range(18)),                         # 동일
}
# junction 인근 bp: 전환점 ±1 범위
JUNCTION_NEIGHBOR = 1
# internal bp: 양쪽 끝단 N개 제외
TERMINAL_EXCLUDE = 1  # bp_0, bp_17 제외

# ============================================================
# DNA chemistry
# ============================================================
BASEPAIR_HEAVY_ATOM_RULES = {
    ("A", "T"): [("N6", "O4"), ("N1", "N3")],
    ("T", "A"): [("O4", "N6"), ("N3", "N1")],
    ("G", "C"): [("O6", "N4"), ("N1", "N3"), ("N2", "O2")],
    ("C", "G"): [("N4", "O6"), ("N3", "N1"), ("O2", "N2")],
}

PHOSPHATE_NAMES = {"P", "OP1", "OP2", "O1P", "O2P"}
BACKBONE_NAMES = {
    "P", "OP1", "OP2", "O1P", "O2P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "O2'",
    "O5*", "C5*", "C4*", "O4*", "C3*", "O3*", "C2*", "C1*", "O2*"
}
WATER_SELECTION = "(resname SOL WAT HOH TIP3 TIP3P) and (name O OW OH2)"
ION_SELECTION = "resname NA NA+ SOD POT K K+ CL CL- MG MG2+ CA CA2+"


# ============================================================
# 유틸: time string → picoseconds 변환
# ============================================================
def parse_time_to_ps(time_str: str) -> float:
    """
    '50ns' -> 50000.0 ps
    '100ps' -> 100.0 ps
    '1us'  -> 1000000.0 ps
    '500'  -> 500.0 ps (단위 없으면 ps로 간주)
    """
    s = time_str.strip().lower()
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*(ns|ps|us|µs)?", s)
    if not m:
        raise ValueError(f"시간 형식을 인식할 수 없습니다: '{time_str}' (예: 50ns, 100ps, 1us)")
    val = float(m.group(1))
    unit = m.group(2) or "ps"
    multipliers = {"ps": 1.0, "ns": 1000.0, "us": 1e6, "µs": 1e6}
    return val * multipliers[unit]


# ============================================================
# [FIX] 안전한 AtomGroup 합산 함수
# ============================================================
def merge_atomgroups(ag_list):
    """
    여러 AtomGroup을 하나로 합칩니다.
    sum(generator, start=u.atoms[:0]) 대신 사용 — MDAnalysis 버전 호환성 문제 해결.
    """
    valid = [ag for ag in ag_list if ag is not None and len(ag) > 0]
    if len(valid) == 0:
        return None  # 호출부에서 None 체크 필요
    if len(valid) == 1:
        return valid[0]
    # mda.Merge는 Universe를 새로 만들어버리므로 단순 concatenate 사용
    import functools, operator
    return functools.reduce(operator.add, valid)


def residues_to_atomgroup(residues, universe):
    """Residue 리스트를 하나의 AtomGroup으로 변환 (안전 버전)"""
    if len(residues) == 0:
        return universe.atoms[:0]
    idx = np.concatenate([res.atoms.indices for res in residues])
    return universe.atoms[idx]


# ============================================================
# 나머지 함수들 (기존과 동일)
# ============================================================
def norm_base(resname: str) -> str:
    rn = resname.upper()
    if "DA" in rn: return "A"
    if "DT" in rn: return "T"
    if "DG" in rn: return "G"
    if "DC" in rn: return "C"
    return rn


def extract_temperature(folder):
    folder = Path(folder)
    name = folder.name

    if re.fullmatch(r"\d{2,4}", name):
        val = int(name)
        if 200 <= val <= 500:
            return val

    m = re.search(r"(?<!\d)(\d{2,4})\s*[Kk](?![A-Za-z])", name)
    if m:
        val = int(m.group(1))
        if 200 <= val <= 500:
            return val

    m = re.search(r"(?<!\d)(\d{3})(?!\d)", name)
    if m:
        val = int(m.group(1))
        if 200 <= val <= 500:
            return val

    logf = folder / LOG_NAME
    if logf.exists():
        txt = logf.read_text(errors="ignore")
        for pat in [r"\bref_t\s*=\s*([0-9]+(?:\.[0-9]+)?)", r"\bgen_temp\s*=\s*([0-9]+(?:\.[0-9]+)?)",
                    r"\bref-t\s*=\s*([0-9]+(?:\.[0-9]+)?)", r"\bgen-temp\s*=\s*([0-9]+(?:\.[0-9]+)?)"]:
            m = re.search(pat, txt, re.IGNORECASE)
            if m:
                val = float(m.group(1))
                if 200 <= val <= 500:
                    return int(round(val))

    return np.nan


def extract_system_name(system_dir):
    system_dir = Path(system_dir)
    root = Path(ROOT_DIR).resolve()
    parent = system_dir.parent.resolve()
    if parent != root:
        return system_dir.parent.name
    return system_dir.name


def logistic(T, low, high, Tm, k):
    return low + (high - low) / (1.0 + np.exp((T - Tm) / k))


def find_system_dirs(root):
    root = Path(root)
    dirs = []
    for gro in root.rglob(TOPO_NAME):
        xtc = gro.parent / TRAJ_NAME
        if xtc.exists():
            dirs.append(gro.parent)
    return sorted(set(dirs))


def get_nucleic_residues_in_order(u):
    nuc = u.select_atoms("nucleic")
    if len(nuc.residues) == 0:
        return list(u.residues)
    return list(nuc.residues)


def get_atom_pos(residue, atom_name):
    ag = residue.atoms.select_atoms(f"name {atom_name}")
    if len(ag) == 0:
        return None
    return ag.positions[0]


def get_bp_pairs_by_order(residues, bp_per_strand=18):
    if len(residues) < 2 * bp_per_strand:
        raise ValueError(f"Need at least {2*bp_per_strand} residues, found {len(residues)}")
    return [(i, 2 * bp_per_strand - 1 - i) for i in range(bp_per_strand)]


def compute_bp_fraction_from_ordered_residues(residues, bp_pairs, cutoff=3.5):
    formed = 0
    valid = 0
    per_bp = {}

    for i, j in bp_pairs:
        r1 = residues[i]
        r2 = residues[j]
        b1 = norm_base(r1.resname)
        b2 = norm_base(r2.resname)
        key = (b1, b2)
        label = f"{i}_{j}_{r1.resname}_{r2.resname}"

        if key not in BASEPAIR_HEAVY_ATOM_RULES:
            per_bp[label] = np.nan
            continue

        ok = True
        for a1, a2 in BASEPAIR_HEAVY_ATOM_RULES[key]:
            p1 = get_atom_pos(r1, a1)
            p2 = get_atom_pos(r2, a2)
            if p1 is None or p2 is None:
                ok = False
                break
            if np.linalg.norm(p1 - p2) > cutoff:
                ok = False
                break

        valid += 1
        per_bp[label] = 1 if ok else 0
        formed += int(ok)

    frac = formed / valid if valid > 0 else np.nan
    return frac, per_bp


def build_native_contacts(strand1_atoms, strand2_atoms, cutoff=4.5):
    ag1 = strand1_atoms.select_atoms("not name H*")
    ag2 = strand2_atoms.select_atoms("not name H*")
    dmat = distance_array(ag1.positions, ag2.positions)
    pairs = np.argwhere(dmat < cutoff)
    native_pairs = [(int(i), int(j)) for i, j in pairs]
    return ag1, ag2, native_pairs


def native_contact_fraction(ag1, ag2, native_pairs, cutoff=4.5):
    if len(native_pairs) == 0:
        return np.nan
    dmat = distance_array(ag1.positions, ag2.positions)
    count = sum(1 for i, j in native_pairs if dmat[i, j] < cutoff)
    return count / len(native_pairs)


def rmsd_superposed(P, Q):
    P = P.astype(float)
    Q = Q.astype(float)
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    C = np.dot(Pc.T, Qc)
    V, _, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(np.dot(V, Wt)))
    D = np.diag([1.0, 1.0, d])
    U = np.dot(np.dot(V, D), Wt)
    P_rot = np.dot(Pc, U)
    diff = P_rot - Qc
    return np.sqrt((diff * diff).sum() / len(P))


def atomgroup_or_empty(u, selections):
    for sel in selections:
        ag = u.select_atoms(sel)
        if len(ag) > 0:
            return ag
    return u.atoms[:0]


def get_phosphate_atom(res):
    for nm in ["P", "OP1", "O1P", "OP2", "O2P"]:
        ag = res.atoms.select_atoms(f"name {nm}")
        if len(ag) > 0:
            return ag[0]
    return None


def get_base_heavy_atoms(res):
    names = [a.name for a in res.atoms if (a.name.upper() not in BACKBONE_NAMES and not a.name.upper().startswith("H"))]
    if len(names) == 0:
        return res.atoms[:0]
    uniq = sorted(set(names))
    return res.atoms.select_atoms("name " + " ".join(uniq))


def residue_phosphate_accessibility_metrics(residues, strand1_res, strand2_res, water_ox, ion_atoms, universe):
    phosphate_atoms = []
    phosphate_labels = []
    per_res_metrics = []

    strand1_ids = {r.ix for r in strand1_res}

    # strand2_res 를 AtomGroup으로 변환 (안전 버전)
    strand2_ag = residues_to_atomgroup(strand2_res, universe)
    strand1_ag = residues_to_atomgroup(strand1_res, universe)

    for idx, res in enumerate(residues):
        p_atom = get_phosphate_atom(res)
        if p_atom is None:
            continue

        opp_heavy_ag = strand2_ag.select_atoms("not name H*") if res.ix in strand1_ids else strand1_ag.select_atoms("not name H*")

        base_pool = []
        for r_other in residues:
            if abs(r_other.resid - res.resid) <= 1:
                continue
            base_pool.extend([a.ix for a in get_base_heavy_atoms(r_other)])
        base_atoms = universe.atoms[np.array(base_pool, dtype=int)] if len(base_pool) else universe.atoms[:0]

        water_contacts = ion_contacts = base_shield = opp_contacts = 0

        if len(water_ox) > 0:
            d = distance_array(p_atom.position[None, :], water_ox.positions)[0]
            water_contacts = int(np.sum(d < WATER_CONTACT_CUTOFF))

        if len(ion_atoms) > 0:
            d = distance_array(p_atom.position[None, :], ion_atoms.positions)[0]
            ion_contacts = int(np.sum(d < WATER_CONTACT_CUTOFF))

        if len(base_atoms) > 0:
            d = distance_array(p_atom.position[None, :], base_atoms.positions)[0]
            base_shield = int(np.sum(d < BASE_SHIELD_CUTOFF))

        if len(opp_heavy_ag) > 0:
            d = distance_array(p_atom.position[None, :], opp_heavy_ag.positions)[0]
            opp_contacts = int(np.sum(d < OPP_STRAND_CONTACT_CUTOFF))

        exposure_score = (water_contacts + 0.5 * ion_contacts + 0.5 * opp_contacts) / (1.0 + base_shield)

        phosphate_atoms.append(p_atom)
        phosphate_labels.append(f"res_{idx}_{res.resname}_{res.resid}")
        per_res_metrics.append({
            "label": phosphate_labels[-1],
            "water_contacts": water_contacts,
            "ion_contacts": ion_contacts,
            "base_shield": base_shield,
            "opp_contacts": opp_contacts,
            "exposure_score": exposure_score,
        })

    return phosphate_atoms, phosphate_labels, per_res_metrics


def get_junction_bp_indices(system_name, bp_per_strand=18):
    """
    시스템 이름을 소문자+특수문자 제거 후 SYSTEM_JUNCTION_BP key와 부분 매칭.
    가장 구체적인(긴) key 우선 매칭.
    예: 'D9L9' -> 'd9l9', 'd3l3' -> 'd3l3', 'all_D' -> 'alld'
    """
    normalized = system_name.lower().replace("_", "").replace("-", "").replace("(", "").replace(")", "").replace(" ", "")
    # 길이 내림차순 → 더 구체적인 패턴 우선
    for key in sorted(SYSTEM_JUNCTION_BP.keys(), key=len, reverse=True):
        if key in normalized:
            return SYSTEM_JUNCTION_BP[key]
    return []


def compute_internal_bp_fraction(per_bp, bp_per_strand=18, exclude=TERMINAL_EXCLUDE):
    """양쪽 끝단 exclude개 bp를 제외한 internal bp fraction."""
    valid_keys = [k for k in per_bp if not np.isnan(per_bp[k])]
    # key 형식: "i_j_resname1_resname2" → i로 끝단 판별
    internal = []
    for k in valid_keys:
        i = int(k.split("_")[0])
        if exclude <= i < (bp_per_strand - exclude):
            internal.append(per_bp[k])
    if len(internal) == 0:
        return np.nan
    return float(np.mean(internal))


def compute_max_contiguous_bp(per_bp, bp_per_strand=18, occ_threshold=0.5):
    """
    occupancy > threshold인 bp가 몇 개 연속으로 이어지는지 최대 길이 반환.
    per_bp는 {label: value} dict (단일 frame 또는 mean).
    """
    # index 순서로 정렬
    occ = {}
    for k, v in per_bp.items():
        if np.isnan(v):
            continue
        i = int(k.split("_")[0])
        occ[i] = v

    if len(occ) == 0:
        return 0

    max_run = 0
    cur_run = 0
    for i in range(bp_per_strand):
        if occ.get(i, 0) > occ_threshold:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    return max_run


def compute_junction_bp_fraction(per_bp, junction_indices, neighbor=JUNCTION_NEIGHBOR):
    """
    junction bp ±neighbor 범위의 bp occupancy 평균.
    junction_indices가 비어있으면 nan 반환.
    """
    if len(junction_indices) == 0:
        return np.nan
    # junction 근방 index 집합
    near = set()
    for ji in junction_indices:
        for delta in range(-neighbor, neighbor + 1):
            near.add(ji + delta)

    vals = []
    for k, v in per_bp.items():
        if np.isnan(v):
            continue
        i = int(k.split("_")[0])
        if i in near:
            vals.append(v)
    return float(np.mean(vals)) if len(vals) > 0 else np.nan


def groove_width_proxy(residues, bp_pairs):
    vals = {}
    all_d = []
    for i, j in bp_pairs:
        p1 = get_phosphate_atom(residues[i])
        p2 = get_phosphate_atom(residues[j])
        label = f"pp_{i}_{j}_{residues[i].resname}_{residues[j].resname}"
        if (p1 is None) or (p2 is None):
            vals[label] = np.nan
            continue
        d = float(np.linalg.norm(p1.position - p2.position))
        vals[label] = min(d, GROOVE_PP_CUTOFF)
        all_d.append(vals[label])
    return (np.mean(all_d) if len(all_d) else np.nan), vals


# ============================================================
# [FIX] 핵심 분석 함수 — AtomGroup 합산 방식 수정
# ============================================================
def analyze_system(system_dir, max_time_ps=None, use_last_fraction=0.5):
    system_dir = Path(system_dir)
    print(f"[INFO] analyzing {system_dir}")

    u = mda.Universe(str(system_dir / TOPO_NAME), str(system_dir / TRAJ_NAME))
    temperature = extract_temperature(system_dir)
    system_name = extract_system_name(system_dir)

    residues = get_nucleic_residues_in_order(u)
    if len(residues) < TOTAL_RES:
        raise ValueError(f"{system_dir}: expected at least {TOTAL_RES} residues, found {len(residues)}")
    residues = residues[:TOTAL_RES]
    bp_pairs = get_bp_pairs_by_order(residues, BP_PER_STRAND)

    strand1_res = residues[:BP_PER_STRAND]
    strand2_res = residues[BP_PER_STRAND:TOTAL_RES]

    # ✅ [FIX] sum() 대신 residues_to_atomgroup() 사용
    strand1_atoms = residues_to_atomgroup(strand1_res, u)
    strand2_atoms = residues_to_atomgroup(strand2_res, u)

    water_ox = atomgroup_or_empty(u, [WATER_SELECTION])
    ion_atoms = atomgroup_or_empty(u, [ION_SELECTION])

    # ── 분석할 frame 범위 결정 ─────────────────────────────
    n_frames = len(u.trajectory)
    all_times = np.array([ts.time for ts in u.trajectory])

    if max_time_ps is not None:
        # --t 로 제한: 0 ~ max_time_ps 구간만
        valid_mask = all_times <= max_time_ps
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            raise ValueError(
                f"{system_dir}: max_time_ps={max_time_ps:.1f} ps 이전 frame이 없습니다 "
                f"(trajectory 최소 시간: {all_times[0]:.1f} ps)"
            )
        # 그 구간에서 뒤쪽 use_last_fraction 사용
        n_valid = len(valid_indices)
        start_idx_in_valid = int((1.0 - use_last_fraction) * n_valid)
        use_indices = valid_indices[start_idx_in_valid::FRAME_STRIDE]
    else:
        # 기존 방식: 전체 trajectory의 뒤쪽 fraction
        start_frame = int((1.0 - use_last_fraction) * n_frames)
        use_indices = list(range(start_frame, n_frames, FRAME_STRIDE))

    if len(use_indices) == 0:
        raise ValueError(f"{system_dir}: 분석할 frame이 없습니다 (stride={FRAME_STRIDE})")

    print(f"[INFO]   frames 총 {n_frames}개 중 {len(use_indices)}개 분석 "
          f"(시간 범위: {all_times[use_indices[0]]:.1f}~{all_times[use_indices[-1]]:.1f} ps)")

    # ── 레퍼런스 frame (첫 번째 사용 frame) ──────────────────
    u.trajectory[use_indices[0]]

    # ✅ [FIX] sum() 대신 residues_to_atomgroup() 사용
    ref_all = residues_to_atomgroup(residues, u)

    junc_res = [residues[idx] for idx in JUNCTION_RESINDICES if idx < len(residues)]
    ref_junc = residues_to_atomgroup(junc_res, u) if junc_res else ref_all

    # phosphate reference
    ref_P_atoms = []
    ref_P_labels = []
    for idx, res in enumerate(residues):
        p = get_phosphate_atom(res)
        if p is not None:
            ref_P_atoms.append(p)
            ref_P_labels.append(f"res_{idx}_{res.resname}_{res.resid}")
    ref_P = u.atoms[np.array([a.ix for a in ref_P_atoms])] if ref_P_atoms else u.atoms[:0]
    ref_P_pos = ref_P.positions.copy() if len(ref_P) else np.zeros((0, 3))

    ref_all_pos = ref_all.positions.copy()
    ref_junc_pos = ref_junc.positions.copy()
    ag1_native, ag2_native, native_pairs = build_native_contacts(strand1_atoms, strand2_atoms, cutoff=NATIVE_CONTACT_CUTOFF)

    # 새 지표용 junction index
    junction_bp_idx = get_junction_bp_indices(system_name)

    frame_rows, bp_rows, proxy_rows, p_traces = [], [], [], []

    for frame_idx in use_indices:
        ts = u.trajectory[frame_idx]
        time_ps = ts.time

        bp_frac, per_bp = compute_bp_fraction_from_ordered_residues(residues, bp_pairs, cutoff=HBOND_DIST_CUTOFF)

        # ── 새 bp 지표 3종 ──────────────────────────────────
        internal_bp_frac   = compute_internal_bp_fraction(per_bp, BP_PER_STRAND, TERMINAL_EXCLUDE)
        max_contig_bp      = compute_max_contiguous_bp(per_bp, BP_PER_STRAND, occ_threshold=0.5)
        junction_bp_frac   = compute_junction_bp_fraction(per_bp, junction_bp_idx, JUNCTION_NEIGHBOR)
        # ────────────────────────────────────────────────────
        q_native = native_contact_fraction(ag1_native, ag2_native, native_pairs, cutoff=NATIVE_CONTACT_CUTOFF)
        com_dist = np.linalg.norm(strand1_atoms.center_of_mass() - strand2_atoms.center_of_mass())
        rmsd_all = rmsd_superposed(ref_all.positions, ref_all_pos)

        # ✅ [FIX] sum() 대신 residues_to_atomgroup() 사용
        mob_junc = residues_to_atomgroup(junc_res, u) if junc_res else ref_all
        rmsd_junc = rmsd_superposed(mob_junc.positions, ref_junc_pos) if len(mob_junc) else rmsd_all

        phosphate_atoms, phosphate_labels, per_res_metrics = residue_phosphate_accessibility_metrics(
            residues, strand1_res, strand2_res, water_ox, ion_atoms, u
        )
        exposure_vals = [x["exposure_score"] for x in per_res_metrics]
        water_vals = [x["water_contacts"] for x in per_res_metrics]
        shield_vals = [x["base_shield"] for x in per_res_metrics]
        opp_vals = [x["opp_contacts"] for x in per_res_metrics]
        groove_mean, groove_per_pair = groove_width_proxy(residues, bp_pairs)

        exposed_fraction = (
            np.mean([x > np.median(exposure_vals) for x in exposure_vals])
            if len(exposure_vals) else np.nan
        )

        frame_rows.append({
            "system": system_name, "folder": str(system_dir), "temperature_K": temperature,
            "time_ps": time_ps, "bp_fraction": bp_frac,
            "internal_bp_fraction": internal_bp_frac,
            "max_contiguous_bp": max_contig_bp,
            "junction_bp_fraction": junction_bp_frac,
            "native_contact_fraction": q_native,
            "strand_com_distance_A": com_dist, "rmsd_all_A": rmsd_all, "rmsd_junction_A": rmsd_junc,
            "phosphate_water_contacts_mean": np.mean(water_vals) if len(water_vals) else np.nan,
            "phosphate_base_shield_mean": np.mean(shield_vals) if len(shield_vals) else np.nan,
            "phosphate_opp_contacts_mean": np.mean(opp_vals) if len(opp_vals) else np.nan,
            "phosphate_exposure_proxy_mean": np.mean(exposure_vals) if len(exposure_vals) else np.nan,
            "phosphate_exposed_fraction": exposed_fraction,
            "groove_pp_distance_mean_A": groove_mean,
        })

        bp_row = {"system": system_name, "folder": str(system_dir), "temperature_K": temperature, "time_ps": time_ps}
        bp_row.update({f"bp_{k}": v for k, v in per_bp.items()})
        bp_rows.append(bp_row)

        proxy_row = {"system": system_name, "folder": str(system_dir), "temperature_K": temperature, "time_ps": time_ps}
        proxy_row.update({f"phos_{m['label']}_exposure": m["exposure_score"] for m in per_res_metrics})
        proxy_row.update({f"phos_{m['label']}_water": m["water_contacts"] for m in per_res_metrics})
        proxy_row.update({f"phos_{m['label']}_shield": m["base_shield"] for m in per_res_metrics})
        proxy_row.update({f"phos_{m['label']}_opp": m["opp_contacts"] for m in per_res_metrics})
        proxy_row.update(groove_per_pair)
        proxy_rows.append(proxy_row)

        if len(ref_P):
            p_traces.append(ref_P.positions.copy())

    frames_df = pd.DataFrame(frame_rows)
    bp_df = pd.DataFrame(bp_rows)
    proxy_df = pd.DataFrame(proxy_rows)

    summary = {
        "system": system_name, "folder": str(system_dir),
        "temperature_K": temperature, "n_frames_used": len(frames_df),
    }

    for metric in [
        "bp_fraction", "internal_bp_fraction", "max_contiguous_bp", "junction_bp_fraction",
        "native_contact_fraction", "strand_com_distance_A", "rmsd_all_A", "rmsd_junction_A",
        "phosphate_water_contacts_mean", "phosphate_base_shield_mean", "phosphate_opp_contacts_mean",
        "phosphate_exposure_proxy_mean", "phosphate_exposed_fraction", "groove_pp_distance_mean_A"
    ]:
        summary[f"{metric}_mean"] = frames_df[metric].mean()
        summary[f"{metric}_std"] = frames_df[metric].std()

    for c in [c for c in bp_df.columns if c.startswith("bp_")]:
        summary[c + "_mean"] = bp_df[c].mean()

    for c in [c for c in proxy_df.columns if c.startswith("phos_") or c.startswith("pp_")]:
        summary[c + "_mean"] = proxy_df[c].mean()

    if len(p_traces) > 1:
        Pstack = np.stack(p_traces, axis=0)
        if Pstack.shape[1] == len(ref_P_labels) and len(ref_P_pos) > 0:
            ref_cent = ref_P_pos.mean(axis=0)
            Pstack_centered = Pstack - Pstack.mean(axis=1, keepdims=True) + ref_cent
            diff = Pstack_centered - ref_P_pos[None, :, :]
            rmsf = np.sqrt(np.mean(np.sum(diff * diff, axis=2), axis=0))
            summary["phosphate_rmsf_mean_A"] = float(np.mean(rmsf))
            summary["phosphate_rmsf_std_A"] = float(np.std(rmsf))
            for lab, val in zip(ref_P_labels, rmsf):
                summary[f"phosphate_rmsf_{lab}_A"] = float(val)
        else:
            summary["phosphate_rmsf_mean_A"] = np.nan
            summary["phosphate_rmsf_std_A"] = np.nan
    else:
        summary["phosphate_rmsf_mean_A"] = np.nan
        summary["phosphate_rmsf_std_A"] = np.nan

    return frames_df, bp_df, proxy_df, summary


# ============================================================
# PLOTTING (기존과 동일)
# ============================================================
def plot_folder_level(frame_df, outdir):
    outdir.mkdir(exist_ok=True, parents=True)
    metrics = [
        "bp_fraction", "native_contact_fraction", "strand_com_distance_A", "rmsd_all_A", "rmsd_junction_A",
        "phosphate_exposure_proxy_mean", "phosphate_water_contacts_mean", "phosphate_base_shield_mean",
        "groove_pp_distance_mean_A"
    ]
    for folder, sub in frame_df.groupby("folder"):
        sub = sub.sort_values("time_ps")
        folder_name = Path(folder).name
        system_name = sub["system"].iloc[0]
        temp = sub["temperature_K"].iloc[0]
        for metric in metrics:
            plt.figure(figsize=(5, 4))
            plt.plot(sub["time_ps"], sub[metric], lw=1.5)
            plt.xlabel("Time (ps)")
            plt.ylabel(metric)
            plt.title(f"{system_name} | {folder_name} | {temp} K")
            plt.tight_layout()
            plt.savefig(outdir / f"{system_name}_{folder_name}_{metric}_timeseries.png", dpi=200)
            plt.close()


def plot_system_level(summary_df, outdir):
    outdir.mkdir(exist_ok=True, parents=True)
    metrics = [
        "bp_fraction_mean", "native_contact_fraction_mean", "strand_com_distance_A_mean", "rmsd_all_A_mean",
        "rmsd_junction_A_mean", "phosphate_exposure_proxy_mean_mean", "phosphate_water_contacts_mean_mean",
        "phosphate_base_shield_mean_mean", "groove_pp_distance_mean_A_mean", "phosphate_rmsf_mean_A"
    ]
    for system, sub in summary_df.groupby("system"):
        sub = sub.sort_values("temperature_K")
        for metric in metrics:
            std_col = metric.replace("_mean", "_std") if metric.endswith("_mean") else None
            yerr = sub[std_col] if (std_col is not None and std_col in sub.columns) else None
            plt.figure(figsize=(5, 4))
            plt.errorbar(sub["temperature_K"], sub[metric], yerr=yerr, marker="o", capsize=3)
            plt.xlabel("Temperature (K)")
            plt.ylabel(metric)
            plt.title(f"{system}: {metric}")
            plt.tight_layout()
            plt.savefig(outdir / f"{system}_{metric}.png", dpi=200)
            plt.close()


def plot_global_overlay(summary_df, outdir):
    outdir.mkdir(exist_ok=True, parents=True)
    metrics = [
        "bp_fraction_mean", "native_contact_fraction_mean", "phosphate_exposure_proxy_mean_mean",
        "phosphate_water_contacts_mean_mean", "phosphate_base_shield_mean_mean", "phosphate_rmsf_mean_A"
    ]
    for metric in metrics:
        plt.figure(figsize=(6, 4.5))
        for system in sorted(summary_df["system"].dropna().unique()):
            sub = summary_df[summary_df["system"] == system].sort_values("temperature_K")
            plt.plot(sub["temperature_K"], sub[metric], marker="o", label=system)
        plt.xlabel("Temperature (K)")
        plt.ylabel(metric)
        plt.title(f"All systems: {metric}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"ALLSYSTEMS_{metric}.png", dpi=200)
        plt.close()


def plot_bp_structural_metrics(summary_df, outdir):
    """
    [NEW] 3종 추가 bp 지표를 한 figure에 비교:
      - internal_bp_fraction (끝단 제외)
      - max_contiguous_bp (연속 bp 최대 길이)
      - junction_bp_fraction (D/L 전환 근방)
    각 지표별 ALLSYSTEMS 비교 플롯 + 시스템별 3-panel subplot
    """
    outdir.mkdir(exist_ok=True, parents=True)

    new_metrics = [
        ("internal_bp_fraction_mean",  "Internal bp fraction\n(끝단 제외)"),
        ("max_contiguous_bp_mean",     "Max contiguous bp\n(연속 bp 최대 길이)"),
        ("junction_bp_fraction_mean",  "Junction bp fraction\n(D/L 전환 근방)"),
    ]

    # ── (1) ALLSYSTEMS 비교: 3개 subplot ──────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    systems = sorted(summary_df["system"].dropna().unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(systems)))

    for ax, (col, label) in zip(axes, new_metrics):
        for sys, color in zip(systems, colors):
            sub = summary_df[summary_df["system"] == sys].sort_values("temperature_K")
            if col not in sub.columns:
                continue
            std_col = col.replace("_mean", "_std")
            yerr = sub[std_col] if std_col in sub.columns else None
            ax.errorbar(sub["temperature_K"], sub[col], yerr=yerr,
                        marker="o", label=sys, color=color, capsize=3)
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=7)

    fig.suptitle("All systems: structural bp metrics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outdir / "ALLSYSTEMS_bp_structural_metrics.png", dpi=200)
    plt.close()

    # ── (2) 기존 bp_fraction vs internal_bp_fraction 비교 ────
    fig, ax = plt.subplots(figsize=(7, 5))
    ls_map = {"bp_fraction_mean": "--", "internal_bp_fraction_mean": "-"}
    label_map = {"bp_fraction_mean": "bp_fraction (all)", "internal_bp_fraction_mean": "internal_bp_fraction"}
    for sys, color in zip(systems, colors):
        sub = summary_df[summary_df["system"] == sys].sort_values("temperature_K")
        for col, ls in ls_map.items():
            if col not in sub.columns:
                continue
            ax.plot(sub["temperature_K"], sub[col], ls=ls, marker="o",
                    color=color, label=f"{sys} ({label_map[col]})" if col == "internal_bp_fraction_mean" else None,
                    alpha=0.7 if col == "bp_fraction_mean" else 1.0)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("bp fraction")
    ax.set_title("bp_fraction vs internal_bp_fraction\n(실선=internal, 점선=전체)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(outdir / "ALLSYSTEMS_bp_fraction_comparison.png", dpi=200)
    plt.close()

    # ── (3) 시스템별 3-panel: 온도 × 3지표 ──────────────────
    for sys in systems:
        sub = summary_df[summary_df["system"] == sys].sort_values("temperature_K")
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (col, label) in zip(axes, new_metrics):
            if col not in sub.columns:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(label)
                continue
            std_col = col.replace("_mean", "_std")
            yerr = sub[std_col] if std_col in sub.columns else None
            ax.errorbar(sub["temperature_K"], sub[col], yerr=yerr,
                        marker="o", color="steelblue", capsize=3)
            ax.set_xlabel("Temperature (K)")
            ax.set_ylabel(label)
            ax.set_title(label)
        fig.suptitle(f"{sys}: structural bp metrics", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(outdir / f"{sys}_bp_structural_metrics.png", dpi=200)
        plt.close()

    # ── (4) 개별 ALLSYSTEMS 파일도 기존과 동일하게 저장 ───────
    for col, label in new_metrics:
        plt.figure(figsize=(6, 4.5))
        for sys, color in zip(systems, colors):
            sub = summary_df[summary_df["system"] == sys].sort_values("temperature_K")
            if col not in sub.columns:
                continue
            plt.plot(sub["temperature_K"], sub[col], marker="o", label=sys, color=color)
        plt.xlabel("Temperature (K)")
        plt.ylabel(label)
        plt.title(f"All systems: {col}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"ALLSYSTEMS_{col}.png", dpi=200)
        plt.close()


def plot_bp_heatmaps(summary_df, outdir):
    outdir.mkdir(exist_ok=True, parents=True)
    for system, sub in summary_df.groupby("system"):
        sub = sub.sort_values("temperature_K")
        bp_cols = [c for c in sub.columns if c.startswith("bp_") and c.endswith("_mean")]
        if len(bp_cols) == 0:
            continue
        mat = np.array([sub[c].values for c in bp_cols])
        plt.figure(figsize=(8, max(4, len(bp_cols) * 0.25)))
        plt.imshow(mat, aspect="auto", origin="lower")
        plt.yticks(np.arange(len(bp_cols)), [c.replace("_mean", "") for c in bp_cols], fontsize=8)
        plt.xticks(np.arange(len(sub)), sub["temperature_K"].tolist())
        plt.xlabel("Temperature (K)")
        plt.ylabel("Base pair")
        plt.title(f"{system}: base-pair occupancy")
        cb = plt.colorbar(); cb.set_label("Occupancy")
        plt.tight_layout()
        plt.savefig(outdir / f"{system}_bp_heatmap.png", dpi=200)
        plt.close()


def plot_proxy_heatmaps(summary_df, outdir):
    outdir.mkdir(exist_ok=True, parents=True)
    for system, sub in summary_df.groupby("system"):
        sub = sub.sort_values("temperature_K")
        phos_cols = [c for c in sub.columns if c.startswith("phos_") and c.endswith("_exposure_mean")]
        if not phos_cols:
            continue
        mat = np.array([sub[c].values for c in phos_cols])
        plt.figure(figsize=(8, max(4, len(phos_cols) * 0.22)))
        plt.imshow(mat, aspect="auto", origin="lower")
        plt.yticks(np.arange(len(phos_cols)), [c.replace("_mean", "") for c in phos_cols], fontsize=7)
        plt.xticks(np.arange(len(sub)), sub["temperature_K"].tolist())
        plt.xlabel("Temperature (K)"); plt.ylabel("Phosphate")
        plt.title(f"{system}: phosphate exposure proxy")
        cb = plt.colorbar(); cb.set_label("Exposure proxy")
        plt.tight_layout()
        plt.savefig(outdir / f"{system}_phosphate_exposure_heatmap.png", dpi=200)
        plt.close()


def plot_tm_fits(summary_df, outdir):
    outdir.mkdir(exist_ok=True, parents=True)
    tm_rows = []
    for system in sorted(summary_df["system"].dropna().unique()):
        for metric in ["bp_fraction_mean", "native_contact_fraction_mean", "phosphate_exposure_proxy_mean_mean"]:
            sub = summary_df[summary_df["system"] == system].copy()
            sub = sub.sort_values("temperature_K").dropna(subset=["temperature_K", metric])
            if len(sub) < 4:
                continue
            x = sub["temperature_K"].values.astype(float)
            y = sub[metric].values.astype(float)
            p0 = [np.min(y), np.max(y), np.median(x), 5.0]
            try:
                popt, _ = curve_fit(logistic, x, y, p0=p0, maxfev=10000)
                low, high, Tm, k = popt
                xx = np.linspace(np.min(x), np.max(x), 300)
                yy = logistic(xx, *popt)
                plt.figure(figsize=(5, 4))
                plt.scatter(x, y, s=50)
                plt.plot(xx, yy, lw=2, label=f"Tm={Tm:.1f} K")
                plt.xlabel("Temperature (K)"); plt.ylabel(metric)
                plt.title(f"{system}: {metric}"); plt.legend(); plt.tight_layout()
                plt.savefig(outdir / f"{system}_{metric}_Tmfit.png", dpi=200)
                plt.close()
                tm_rows.append({"system": system, "metric": metric, "Tm_K": Tm, "low": low, "high": high, "k": k})
            except Exception:
                pass
    if tm_rows:
        pd.DataFrame(tm_rows).to_csv(outdir / "Tm_fit_results.csv", index=False)


# ============================================================
# CLI 파싱
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="DNA D/L nuclease proxy analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python dna_nuclease_proxy_analysis_fixed.py
  python dna_nuclease_proxy_analysis_fixed.py --t=50ns
  python dna_nuclease_proxy_analysis_fixed.py --t=100ps
  python dna_nuclease_proxy_analysis_fixed.py --t=1us --last-fraction=0.3
  python dna_nuclease_proxy_analysis_fixed.py --root=./simulations --outdir=./results
        """
    )
    parser.add_argument(
        "--t", dest="max_time", default=None,
        help="분석할 최대 시뮬레이션 시간 (예: 50ns, 100ps, 1us). 미지정 시 전체 trajectory 사용."
    )
    parser.add_argument(
        "--last-fraction", dest="last_fraction", type=float, default=0.5,
        help="지정한 시간 범위에서 뒤쪽 몇 %%를 사용할지 (기본: 0.5 = 50%%)"
    )
    parser.add_argument(
        "--root", default=".",
        help="시뮬레이션 루트 디렉토리 (기본: 현재 디렉토리)"
    )
    parser.add_argument(
        "--outdir", default="analysis_out",
        help="결과 저장 디렉토리 (기본: analysis_out)"
    )
    parser.add_argument(
        "--stride", type=int, default=10,
        help="frame stride (기본: 10)"
    )
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================
def main():
    global ROOT_DIR, OUTDIR, FRAME_STRIDE

    args = parse_args()
    ROOT_DIR = args.root
    OUTDIR = Path(args.outdir)
    OUTDIR.mkdir(exist_ok=True)
    FRAME_STRIDE = args.stride

    max_time_ps = None
    if args.max_time is not None:
        max_time_ps = parse_time_to_ps(args.max_time)
        print(f"[INFO] 분석 시간 상한: {args.max_time} = {max_time_ps:.1f} ps")

    use_last_fraction = args.last_fraction
    print(f"[INFO] 사용 fraction: 뒤쪽 {use_last_fraction*100:.0f}%")

    system_dirs = find_system_dirs(ROOT_DIR)
    if len(system_dirs) == 0:
        raise RuntimeError("No pro.gro/pro.xtc pairs found.")

    print("[INFO] Found directories:")
    for sd in system_dirs:
        print("   ", sd, "-> system:", extract_system_name(sd), "| T:", extract_temperature(sd))

    all_frames, all_bp, all_proxy, all_summary = [], [], [], []
    for sd in system_dirs:
        try:
            frames_df, bp_df, proxy_df, summary = analyze_system(
                sd, max_time_ps=max_time_ps, use_last_fraction=use_last_fraction
            )
            all_frames.append(frames_df)
            all_bp.append(bp_df)
            all_proxy.append(proxy_df)
            all_summary.append(summary)
        except Exception as e:
            print(f"[ERROR] {sd}: {e}")

    if len(all_summary) == 0:
        raise RuntimeError("No systems analyzed successfully.")

    frames_all = pd.concat(all_frames, ignore_index=True)
    bp_all = pd.concat(all_bp, ignore_index=True)
    proxy_all = pd.concat(all_proxy, ignore_index=True)
    summary_df = pd.DataFrame(all_summary)

    frames_all.to_csv(OUTDIR / "all_frames_metrics.csv", index=False)
    bp_all.to_csv(OUTDIR / "all_frames_bp.csv", index=False)
    proxy_all.to_csv(OUTDIR / "all_frames_proxy.csv", index=False)
    summary_df.to_csv(OUTDIR / "summary_by_system_temperature.csv", index=False)

    plot_folder_level(frames_all, OUTDIR / "folder_level")
    plot_system_level(summary_df, OUTDIR / "system_level")
    plot_global_overlay(summary_df, OUTDIR / "global_overlay")
    plot_bp_heatmaps(summary_df, OUTDIR / "bp_heatmaps")
    plot_proxy_heatmaps(summary_df, OUTDIR / "proxy_heatmaps")
    plot_tm_fits(summary_df, OUTDIR / "tm_fits")
    plot_bp_structural_metrics(summary_df, OUTDIR / "bp_structural_metrics")  # [NEW]

    config = {
        "ROOT_DIR": ROOT_DIR,
        "max_time_input": args.max_time,
        "max_time_ps": max_time_ps,
        "use_last_fraction": use_last_fraction,
        "FRAME_STRIDE": FRAME_STRIDE,
        "BP_PER_STRAND": BP_PER_STRAND,
        "TOTAL_RES": TOTAL_RES,
        "HBOND_DIST_CUTOFF": HBOND_DIST_CUTOFF,
        "NATIVE_CONTACT_CUTOFF": NATIVE_CONTACT_CUTOFF,
        "WATER_CONTACT_CUTOFF": WATER_CONTACT_CUTOFF,
        "BASE_SHIELD_CUTOFF": BASE_SHIELD_CUTOFF,
        "OPP_STRAND_CONTACT_CUTOFF": OPP_STRAND_CONTACT_CUTOFF,
        "JUNCTION_RESINDICES": JUNCTION_RESINDICES,
    }
    with open(OUTDIR / "analysis_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[DONE] Results saved to {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
