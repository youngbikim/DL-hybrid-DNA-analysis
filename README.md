# DL-hybrid-DNA-analysis
Python analysis scripts accompanying the D/L hybrid DNA duplex study.

---

## Overview

This repository contains custom Python scripts used to analyze molecular dynamics simulations of stereochemically patterned D/L DNA duplexes.

Implemented analyses include

- Base-pair occupancy
- Internal base-pair fraction
- Junction base-pair fraction
- Native contact fraction
- Groove width proxy
- Phosphate exposure proxy
- Base stacking analysis
- Strand interaction energy analysis

---

## Requirements

Python >=3.10

MDAnalysis
NumPy
SciPy
Pandas
Matplotlib

---

## Usage

### Duplex analysis

python duplex_analysis.py \
    --t 100ns \
    --last-fraction 0.4

### Stacking analysis

python stacking_analysis.py

### Plotting

python plot_stacking.py
