# Pore-Morphology-Method (PMM)

This repository contains the **Pore-Morphology Method (PMM)** to simulate drainage process in porous media. For detailed methodology explanation see:

> Tavakkoli, O., Ebadi, M., Da Wang, Y., Mostaghimi, P., Armstrong, R.T., 2025. "Assessment of wetting conditions in quasistatic drainage modeling using a pore morphology method and j-function wettability estimator." *International Journal of Multiphase Flow* **183**:105067. [doi:10.1016/j.ijmultiphaseflow.2024.105067](https://doi.org/10.1016/j.ijmultiphaseflow.2024.105067)

The code generates capillary-pressure curves by performing morphological operation on 3D segmented micro-CT images, using an adaptive, parallel kernel-size sweep and trapped wetting phase identification.

---

## Features

* Parallel processing with automatic halo handling (split-volume strategy).
* Adaptive kernel search to reach a target (starting) wetting-phase saturation.
* Support for starting kernel runs when a search is not required.
* Trapped wetting phase detection following the algorithm in the paper.

---

## Installation
1) Python 3.9+ recommended.
2) Install dependencies:
- **pip**:

```bash
pip install -r requirements.txt
```

- **conda**:

```bash
conda create -n pmm -c conda-forge python=3.11 numpy scikit-image matplotlib seaborn -y
conda activate pmm
```

---

## Input files

| File | Purpose |
|------|---------|
| **`domain.raw`** | 3D binary image (uint8) of the porous medium:<br>0 = pore, 1 = solid. |
| **`input.txt`** | Run-time parameters (plain text `key = value`). Default template is committed. |

---

## Configuration (`input.txt`)

Example:

```
filename       = domain.raw        # path to the segmented micro-ct image
filesize_x     = 751               # voxels
filesize_y     = 751
filesize_z     = 572
resolution     = 8.42              # µm per voxel
sigma          = 26                # mN/m
theta          = 0                 # degrees
num_threads    = 12                # CPU cores to use

# Kernel–search options
kernel_search  = true              # true: search, false: starting kernel
starting_sat   = 0.95              # starting saturation if search is enabled
starting_kernel = 20               # used only if kernel_search = false
visualization  = true              # true: save .RAW fluid distribution per saturation step, false: skip saving
```

---

## Running

```bash
python PMM.py
```

---

## Output files

| File | Description |
|------|-------------|
| `result_sat<sat>.raw` | Fluid distribution (uint8) for each saturation step (if `visualization = true`). |
| `saturation_vs_pc.pdf` | Capillary pressure curve generated at the end of the run. |
| `result.txt` | Tab-separated values of capillary pressure (Pa) and corresponding saturation. |

---

## Re-using the code

If you use this implementation in academic work, please cite the paper above. A BibTeX entry is:

```bibtex
@article{Tavakkoli2025,
  title = {Assessment of wetting conditions in quasistatic drainage modeling using a pore morphology method and J-function wettability estimator},
  volume = {183},
  ISSN = {0301-9322},
  url = {http://dx.doi.org/10.1016/j.ijmultiphaseflow.2024.105067},
  DOI = {10.1016/j.ijmultiphaseflow.2024.105067},
  journal = {International Journal of Multiphase Flow},
  publisher = {Elsevier BV},
  author = {Tavakkoli,  Omid and Ebadi,  Mohammad and Da Wang,  Ying and Mostaghimi,  Peyman and Armstrong,  Ryan T.},
  year = {2025},
  month = feb,
  pages = {105067}
}
```

---
