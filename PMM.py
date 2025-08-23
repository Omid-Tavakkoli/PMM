import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from skimage import measure
import skimage as ski
import math
import os
import multiprocessing as mp
import sys
from typing import Tuple

sns.set(style="white")


def read_input_params(path="input.txt"):
    """Parse a simple key = value text file and return a lowercase-key dictionary."""
    params = {}
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                # Remove inline comments after the value (everything after '#')
                value = value.split("#", 1)[0].strip()
                params[key.strip().lower()] = value
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file '{path}' not found. The script requires this file with all parameters.") from e
    return params

def _validate_and_read_input():
    """Read params and raw volume, validate sizes; return a dict of runtime inputs."""
    params = read_input_params()

    required_keys = [
        "filename",
        "filesize_x",
        "filesize_y",
        "filesize_z",
        "resolution",
        "sigma",
        "theta",
    ]
    missing = [k for k in required_keys if k not in params]
    if missing:
        raise ValueError(
            f"Missing required keys in input.txt: {', '.join(missing)}"
        )

    file_path = params["filename"]
    width = int(params["filesize_x"])   # x = width
    height = int(params["filesize_y"])  # y = height
    depth = int(params["filesize_z"])   # z = depth

    resolution = float(params["resolution"])   # micron per voxel
    sigma = float(params["sigma"])             # mN/m
    theta = float(params["theta"])             # degrees

    total_elements = depth * height * width
    file_size_bytes = os.path.getsize(file_path)
    expected_bytes = total_elements  # uint8 → 1 byte per voxel
    if file_size_bytes != expected_bytes:
        raise ValueError(
            "Image dimension error: 'filesize_x/y/z' in input.txt do not match the actual RAW file size. "
            f"Expected {expected_bytes} bytes, got {file_size_bytes} bytes."
        )

    with open(file_path, "rb") as f:
        raw_data = np.fromfile(f, dtype=np.uint8, count=total_elements)

    porosity = np.sum(raw_data == 0) / (np.sum(raw_data == 0) + np.sum(raw_data == 1))
    domain_full = raw_data.reshape((depth, height, width))
    domain_full[domain_full == 2] = 1  # normalize if present

    return {
        "params": params,
        "file_path": file_path,
        "width": width,
        "height": height,
        "depth": depth,
        "resolution": resolution,
        "sigma": sigma,
        "theta": theta,
        "total_elements": total_elements,
        "porosity": porosity,
        "domain_full": domain_full,
    }

# --------------------------------------------------------------------------------------
# Fast structuring-element cache and multi-pass morphology utilities
# --------------------------------------------------------------------------------------

MAX_SE_RADIUS = 10  # radii above this threshold are decomposed into multi-pass ops
_SE_CACHE: dict[int, np.ndarray] = {}

def _get_ball(radius: int) -> np.ndarray:
    """Return a cached spherical structuring element of *radius*."""
    if radius <= 0:
        raise ValueError("Structuring-element radius must be positive")
    ball = _SE_CACHE.get(radius)
    if ball is None:
        ball = ski.morphology.ball(radius, dtype=np.uint8)
        _SE_CACHE[radius] = ball
    return ball


def _morphology_multi_pass(volume: np.ndarray, total_radius: int, op: str) -> np.ndarray:
    """Binary dilation/erosion with large *total_radius* via chained passes."""
    if total_radius <= 0:
        return volume
    remaining = total_radius
    result = volume
    while remaining > 0:
        step = min(remaining, MAX_SE_RADIUS)
        se = _get_ball(step)
        if op == "dilation":
            result = ski.morphology.binary_dilation(result, footprint=se)
        elif op == "erosion":
            result = ski.morphology.binary_erosion(result, footprint=se)
        else:
            raise ValueError(f"Unsupported morphology op: {op}")
        remaining -= step
    return result

# --------------------------------------------------------------------------------------
# Saturation computation rules
# --------------------------------------------------------------------------------------

def compute_saturation(
    domain: np.ndarray,
    k: int,
    theta: float,
    mask: np.ndarray | None = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Return (saturation, comb, new_mask) on *domain* with kernel *k*.

    """
    theta_rad = math.radians(theta)
    kernel_size_solid = max(round(k * math.cos(theta_rad)), 1)
    kernel_size_nwp   = round(k)

    # 1) Grain dilation (solid)
    grain_dilation = _morphology_multi_pass(domain, kernel_size_solid, "dilation")

    # 2) Connectivity analysis
    NW_reservoir = np.zeros((1, domain.shape[1], domain.shape[2]), dtype=np.uint8)
    ready_pore_labels = np.concatenate((NW_reservoir, grain_dilation), axis=0)
    pore_labels = measure.label(ready_pore_labels, background=1)
    value_connected_pores = np.unique(pore_labels[0, :, :])

    pore_labels[(pore_labels != 0) & (pore_labels != value_connected_pores)] = 10
    pore_labels[pore_labels == value_connected_pores] = 127
    pore_labels[pore_labels == 0] = 2
    pore_labels[pore_labels == 10] = 1
    pore_labels[pore_labels == 127] = 0
    pore_labels[pore_labels == 2] = 1

    ready_for_nwd = pore_labels[1 : domain.shape[0] + 1]

    # 3) Non-wetting phase erosion
    nwp_dilation = _morphology_multi_pass(ready_for_nwd, kernel_size_nwp, "erosion")

    # 4) Combine phases
    comb = np.copy(domain)
    comb[comb == 0] = 3
    comb[comb == 1] = 2
    comb[comb == 3] = 1
    comb[nwp_dilation == 0] = 0
    comb[domain == 1] = 2

    # ------------------------------------------------------------------
    # Identify trapped non-wetting phase and apply to comb
    # ------------------------------------------------------------------
    wp_reservoir = np.ones((1, domain.shape[1], domain.shape[2]), dtype=np.uint8)
    ready_pore_labels_2 = np.concatenate((comb, wp_reservoir), axis=0)
    ready_pore_labels_2[ready_pore_labels_2 == 0] = 2  # treat NWP as background for connectivity

    pore_labels_2 = measure.label(ready_pore_labels_2, background=2)
    value_connected_pores_2 = np.unique(pore_labels_2[-1, :, :])  # connected to top reservoir slice

    interior_labels = pore_labels_2[:-1]
    trapped = np.ones_like(domain, dtype=np.uint8)
    trapped[(interior_labels != 0) & (interior_labels != value_connected_pores_2)] = 0

    comb[trapped == 0] = 1 

    # Saturation
    wp = np.sum(comb == 1)
    nwp = np.sum(comb == 0)
    saturation = 1 - (nwp / (wp + nwp))

    return saturation, comb, trapped

# --------------------------------------------------------------------------------------
# Parallel morphology helpers – split volume along z, stitch results
# --------------------------------------------------------------------------------------


def _apply_morphology_chunk(args):
    """Worker that applies binary morphology (dilation/erosion) to a sub-volume with halo."""
    volume, radius, op, z_start, z_end = args

    halo = radius
    z0 = max(0, z_start - halo)
    z1 = min(volume.shape[0], z_end + halo)

    sub_vol = volume[z0:z1]
    processed_sub = _morphology_multi_pass(sub_vol, radius, op)

    crop_from = z_start - z0
    crop_to = crop_from + (z_end - z_start)
    return processed_sub[crop_from:crop_to]


def _morphology_parallel(volume: np.ndarray, radius: int, op: str, num_workers: int) -> np.ndarray:
    """Apply *op* ('dilation' or 'erosion') with *radius* across *volume* in parallel."""
    if num_workers < 2 or volume.shape[0] < 2:
        return _morphology_multi_pass(volume, radius, op)

    z_dim = volume.shape[0]
    chunk_sz = math.ceil(z_dim / num_workers)
    tasks = []
    for i in range(num_workers):
        z_start = i * chunk_sz
        z_end = min(z_dim, (i + 1) * chunk_sz)
        if z_start >= z_dim:
            break
        tasks.append((volume, radius, op, z_start, z_end))

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_apply_morphology_chunk, tasks)

    result = np.empty_like(volume, dtype=np.uint8)
    for task, chunk_out in zip(tasks, results):
        z_start = task[3]
        result[z_start : z_start + chunk_out.shape[0]] = chunk_out

    return result


def compute_saturation_parallel(
    domain: np.ndarray,
    k: int,
    theta: float,
    num_workers: int,
    mask: np.ndarray | None = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute saturation and return (sat, comb, new_mask).

    """
    theta_rad = math.radians(theta)
    r_solid = max(round(k * math.cos(theta_rad)), 1)
    r_nwp = round(k)

    # 1) Grain dilation (parallel)
    grain_dilation = _morphology_parallel(domain, r_solid, "dilation", num_workers)

    # 2) Connectivity to non-wetting reservoir (single global labeling)
    NW_reservoir = np.zeros((1, domain.shape[1], domain.shape[2]), dtype=np.uint8)
    ready_pore_labels = np.concatenate((NW_reservoir, grain_dilation), axis=0)
    pore_labels = measure.label(ready_pore_labels, background=1)
    value_connected_pores = np.unique(pore_labels[0, :, :])

    pore_labels[(pore_labels != 0) & (pore_labels != value_connected_pores)] = 10
    pore_labels[pore_labels == value_connected_pores] = 127
    pore_labels[pore_labels == 0] = 2
    pore_labels[pore_labels == 10] = 1
    pore_labels[pore_labels == 127] = 0
    pore_labels[pore_labels == 2] = 1

    ready_for_nwd = pore_labels[1: domain.shape[0] + 1]

    # 3) Non-wetting erosion (parallel)
    nwp_dilation = _morphology_parallel(ready_for_nwd, r_nwp, "erosion", num_workers)

    # 4) Combine phases (same as compute_saturation)
    comb = np.copy(domain)
    comb[comb == 0] = 3
    comb[comb == 1] = 2
    comb[comb == 3] = 1
    comb[nwp_dilation == 0] = 0
    comb[domain == 1] = 2

    # Apply previous trapped mask
    if mask is not None:
        comb[mask == 0] = 1

    # 5) Trapped WP
    wp_reservoir = np.ones((1, domain.shape[1], domain.shape[2]), dtype=np.uint8)
    ready_pore_labels_2 = np.concatenate((comb, wp_reservoir), axis=0)
    ready_pore_labels_2[ready_pore_labels_2 == 0] = 2
    pore_labels_2 = measure.label(ready_pore_labels_2, background=2)
    value_connected_pores_2 = np.unique(pore_labels_2[-1, :, :])
    interior_labels = pore_labels_2[:-1]
    trapped = np.ones_like(domain, dtype=np.uint8)
    trapped[(interior_labels != 0) & (interior_labels != value_connected_pores_2)] = 0
    comb[trapped == 0] = 1  # Apply trapped WP

    wp = np.sum(comb == 1)
    nwp = np.sum(comb == 0)
    saturation = 1 - (nwp / (wp + nwp))

    return saturation, comb, trapped

# --------------------------------------------------------------------------------------
# Kernel-size search utilities
# --------------------------------------------------------------------------------------

def find_best_kernel_size(
    domain_test: np.ndarray,
    target_sat: float,
    theta: float,
    num_workers: int,
    initial_k: int = 20,
    tol: float = 1e-4,
    max_kernel_size: int = 200,
):
    """Search for kernel size yielding *target_sat* starting from *initial_k*."""
    k = initial_k
    step = 1  # initialize step to a positive value for first iteration
    best_k = k
    best_diff = float("inf")
    tested = set()

    while k <= max_kernel_size:
        sat, _, _ = compute_saturation_parallel(domain_test, k, theta, num_workers)
        tested.add(k)
        diff = abs(sat - target_sat)
        print(f"Kernel {k} -> sat {sat:.4f} (diff {diff:.4f})")

        # ------------------------------------------------------------------
        # Determine step to take *before* deciding on overshoot refinement
        # ------------------------------------------------------------------
        gap = target_sat - sat

        def _step_from_abs(g: float) -> int:
            if g > 0.95:
                return 20
            elif g > 0.8:
                return 16
            elif g > 0.6:
                return 14
            elif g > 0.5:
                return 12
            elif g > 0.4:
                return 10
            elif g > 0.3:
                return 8
            elif g > 0.2:
                return 6
            elif g > 0.1:
                return 4
            elif g > 0.05:
                return 2
            else:
                return 1

        step_mag = _step_from_abs(abs(gap))
        step_candidate = step_mag if gap > 0 else -step_mag

        # ------------------------------------------------------------------
        # Update best diff
        # ------------------------------------------------------------------
        if diff <= tol:
            best_k = k
            best_diff = diff
            break
        if diff < best_diff:
            best_diff = diff
            best_k = k

        # ------------------------------------------------------------------
        # Overshoot refinement
        # ------------------------------------------------------------------
        if sat >= target_sat and step_candidate > 0:
            # Downward scan
            for kk in range(k - 1, 0, -1):
                if kk in tested:
                    continue
                sat_k, _, _ = compute_saturation_parallel(domain_test, kk, theta, num_workers)
                diff_k = abs(sat_k - target_sat)
                print(f"  Downward {kk} -> sat {sat_k:.4f} (diff {diff_k:.4f})")
                if diff_k < best_diff:
                    best_diff = diff_k
                    best_k = kk
                if diff_k > best_diff:
                    break
            break
        elif sat <= target_sat and step_candidate < 0:
            # Upward scan (symmetric logic)
            consecutive_worse = 0
            prev_diff = diff
            for k_up in range(k + 1, max_kernel_size + 1):
                if k_up in tested:
                    continue
                sat_up, _, _ = compute_saturation_parallel(domain_test, k_up, theta, num_workers)
                tested.add(k_up)
                diff_up = abs(sat_up - target_sat)
                print(f"  Upward   {k_up} -> sat {sat_up:.4f} (diff {diff_up:.4f})")

                if diff_up < best_diff:
                    best_diff = diff_up
                    best_k = k_up

                if diff_up > prev_diff:
                    consecutive_worse += 1
                    if consecutive_worse >= 2:
                        print("Stopping upward search")
                        break
                else:
                    consecutive_worse = 0

                prev_diff = diff_up
            break
        # Advance kernel size with oscillation and bounds guard
        next_k = k + step_candidate
        if next_k in tested:
            print("Stopping search (oscillation detected).")
            break
        if next_k < 1 or next_k > max_kernel_size:
            print("Stopping search (out of bounds).")
            break
        step = step_candidate
        k = next_k
    return best_k, best_diff

def main():
    # Read configuration and input data
    runtime = _validate_and_read_input()
    params = runtime["params"]
    file_path = runtime["file_path"]
    width = runtime["width"]
    height = runtime["height"]
    depth = runtime["depth"]
    resolution = runtime["resolution"]
    sigma = runtime["sigma"]
    theta = runtime["theta"]
    porosity = runtime["porosity"]
    domain_full = runtime["domain_full"]

    kernel_search_flag = params.get("kernel_search", "true").strip().lower() == "true"
    starting_kernel_param = params.get("starting_kernel")
    starting_sat_param = params.get("starting_sat")
    visualization_flag = params.get("visualization", "true").strip().lower() == "true"

    num_workers = int(params.get("num_threads", mp.cpu_count()))

    print("input parameters loaded from input.txt")
    print("Input Parameters:")
    print(f"Input file: {file_path}")
    print(f"File size: {width} x {height} x {depth}")
    print(f"IFT (sigma): {sigma} mN/m")
    print(f"Contact angle (theta): {theta} degrees")
    print(f"Image resolution: {resolution} micro meter")
    print(f"Num threads = {params.get('num_threads', 'auto')}")
    print("====================================")
    print("Image loaded successfully")
    print(f"Porous domain porosity: {porosity:.4f}")
    print("=====================================")

    # Decide initial kernel size
    if kernel_search_flag:
        if starting_sat_param is None:
            raise ValueError("'starting_sat' must be provided in input.txt when kernel_search = true")
        target_sat = float(starting_sat_param)
        print(f"Finding kernel size to start the simulation from saturation {target_sat} ...")
        k_best, diff_best = find_best_kernel_size(domain_full, target_sat, theta, num_workers, initial_k=20)
        print("=" * 50)
        print(f"Chosen kernel size to start the simulation: {k_best} (diff {diff_best:.4f})")
    else:
        if starting_kernel_param is None:
            raise ValueError("'starting_kernel' must be provided in input.txt when kernel_search = false")
        k_best = int(starting_kernel_param)
        print(f"Skipping kernel search. Simualting kernel size = {k_best} on full domain")
        print("=" * 55)

    # Apply kernel size to full domain
    mask_global = np.ones_like(domain_full, dtype=np.uint8)
    sat_full, comb_full, mask_global = compute_saturation_parallel(domain_full, k_best, theta, num_workers, mask_global)
    print(f"Starting saturation: {sat_full:.4f}")

    # Lists for plotting
    kernel_list: list[int] = [k_best]
    sat_list: list[float] = [sat_full]

    if visualization_flag:
        out_name = f"result_sat{sat_full:.4f}.raw"
        comb_full.astype(np.uint8).tofile(out_name)
        print(f"Saved result to {out_name}")

    # Decrease kernel size until saturation no longer changes
    current_k = k_best
    prev_sat = sat_full
    consecutive_small_changes = 0
    while current_k > 1:
        next_k = current_k - 1
        sat_next, comb_next, mask_global = compute_saturation_parallel(domain_full, next_k, theta, num_workers, mask_global)
        if visualization_flag:
            out_name = f"result_sat{sat_next:.4f}.raw"
            comb_next.astype(np.uint8).tofile(out_name)
            print(f"Kernel {next_k} -> sat {sat_next:.4f} (saved {out_name})")
        else:
            print(f"Kernel {next_k} -> sat {sat_next:.4f}")

        kernel_list.append(next_k)
        sat_list.append(sat_next)

        # Check convergence condition
        sat_change = abs(sat_next - prev_sat)
        if sat_change < 1e-6:
            consecutive_small_changes += 1
        else:
            consecutive_small_changes = 0
            
        # Break only if we have 3 consecutive small changes AND saturation is not 1.0
        if consecutive_small_changes >= 3 and sat_next < 1.0:
            break

        current_k = next_k
        prev_sat = sat_next
        comb_full = comb_next

    # Plot saturation vs kernel size and save
    theta_rad = math.radians(theta)
    pc_list = [(2 * sigma * math.cos(theta_rad) * 1000) / (k * resolution) for k in kernel_list]

    plt.figure()
    plt.plot(sat_list, pc_list, marker="o")
    plt.xlabel("$S_w$")
    plt.ylabel("Capillary pressure (Pa)")
    plt.title("Capillary Pressure vs Wetting Phase Saturation")
    plt.tight_layout()
    plot_name = "saturation_vs_pc.pdf"
    plt.savefig(plot_name, dpi=300)
    print(f"Plot saved to {plot_name}")

    # Write numeric results to text file (Pc  Saturation)
    with open("result.txt", "w") as fp:
        fp.write("input parameters loaded from input.txt\n")
        fp.write("Input Parameters:\n")
        fp.write(f"Input file: {file_path}\n")
        fp.write(f"File size: {width} x {height} x {depth}\n")
        fp.write(f"IFT (sigma): {sigma} mN/m\n")
        fp.write(f"Contact angle (theta): {theta} degrees\n")
        fp.write(f"Image resolution: {resolution} micro meter\n")
        fp.write(f"Num threads = {num_workers}\n")
        fp.write("====================================\n")
        fp.write(f"Porous domain porosity: {porosity:.4f}\n")
        fp.write("====================================\n")
        fp.write("Pc(Pa)\t\tSw\n")
        for pc, sw in zip(pc_list, sat_list):
            fp.write(f"{pc:.6f}\t\t{sw:.6f}\n")
        print("Numeric results saved to result.txt")

    plt.show()
    print("Done.")


if __name__ == "__main__":
    # Select a cross-platform start method for multiprocessing
    if sys.platform.startswith("win"):
        # Windows only supports 'spawn'
        mp.set_start_method("spawn", force=True)
    else:
        # On POSIX (Linux/macOS), keep the interpreter default unless user overrides
        # This avoids issues on recent macOS where 'spawn' is default and recommended
        pass

    main()

