import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_snapshots(input_dir): # '/home/shengqin/lys/code/BOFA/logs/bofa/imagenetr/0/20/seed_1993_model_bofa_20260505_102348/b_analysis/task_0i_b_snapshot.pt'
    snapshot_paths = sorted(glob.glob(os.path.join(input_dir, "task_*_b_snapshot.pt"))) # 所有任务的 pt 文件
    # if not snapshot_paths:
    #     raise FileNotFoundError(f"No task_*_b_snapshot.pt found under {input_dir}")
    return [torch.load(path, map_location="cpu") for path in snapshot_paths]


def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def padded_col_norm_matrix(arrays):
    max_cols = max(arr.shape[1] for arr in arrays)
    heatmap = np.full((len(arrays), max_cols), np.nan, dtype=np.float32)
    for i, arr in enumerate(arrays):
        norms = np.linalg.norm(arr, axis=0)
        heatmap[i, : len(norms)] = norms
    return heatmap


def plot_line(x, ys, labels, ylabel, title, output_path):
    plt.figure(figsize=(8, 4))
    for y, label in zip(ys, labels):
        plt.plot(x, y, marker="o", label=label)
    plt.xlabel("Task")
    plt.ylabel(ylabel)
    plt.title(title)
    if len(labels) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_heatmap(matrix, title, output_path, xlabel="Direction", ylabel="Task"):
    plt.figure(figsize=(10, 4))
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")
    plt.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap)
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_param_heatmap(matrix, title, output_path):
    values = np.abs(matrix)
    vmax = float(np.percentile(values, 99)) if values.size > 0 else 1.0
    plt.figure(figsize=(10, 6))
    plt.imshow(values, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0, vmax=max(vmax, 1e-8))
    plt.colorbar()
    plt.xlabel("Direction")
    plt.ylabel("Output dim")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_svd_spectra(arrays, title, output_path, topk=20):
    plt.figure(figsize=(8, 4)) # 20
    for task_id, arr in enumerate(arrays): #
        singular_values = np.linalg.svd(arr, compute_uv=False)[:topk] # [:topk] 只取前20个最大的奇异值
        plt.plot(np.arange(1, len(singular_values) + 1), singular_values, marker="o", label=f"task_{task_id}")
    plt.xlabel("Singular value index")
    plt.ylabel("Singular value")
    plt.title(title)
    if len(arrays) <= 10:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_metrics_json(metrics, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

# Bs 10ge [task0:(512,498),taski+1:(512,30)]
def analyze_single_b(snapshots, output_dir, with_svd=True):
    Bs = [to_numpy(snapshot["B"]) for snapshot in snapshots]
    task_ids = [int(snapshot["task_id"]) for snapshot in snapshots] # 1-10
    # 大的 Frobenius 范数：B 矩阵的权重变化大
    fro_norms = [float(np.linalg.norm(B)) for B in Bs] # 每个任务的Frobenius 范数 (fro_norms)
    mean_abs = [float(np.abs(B).mean()) for B in Bs] # 平均绝对值 (mean_abs),矩阵元素的平均幅度
    max_abs = [float(np.abs(B).max()) for B in Bs] # 最大绝对值 (max_abs) 衡量权重的"峰值"
    metrics = {
        "task_ids": task_ids,
        "fro_norms": fro_norms,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
    }
    # 通用的折线图绘制函数,绘制一个或多个指标随任务 ID 变化的折线图 b_fro_norm_curve.png b_svd_spectra.png
    plot_line(task_ids, [fro_norms], ["B"], "Frobenius norm", "B Frobenius Norm by Task",
              os.path.join(output_dir, "b_fro_norm_curve.png"))
    plot_heatmap(padded_col_norm_matrix(Bs), "B Column Norm Heatmap", os.path.join(output_dir, "b_col_norm_heatmap.png"))

    for task_id, B in zip(task_ids, Bs):
        plot_param_heatmap(B, f"B Heatmap Task {task_id}", os.path.join(output_dir, f"task_{task_id:02d}_B_heatmap.png"))

    if with_svd: # 取前20个最大的奇异值
        plot_svd_spectra(Bs, "B Singular Value Spectra", os.path.join(output_dir, "b_svd_spectra.png"))

    save_metrics_json(metrics, os.path.join(output_dir, "b_metrics.json"))


def analyze_shared_core(snapshots, output_dir, with_svd=True):
    B_shared_list = [to_numpy(snapshot["B_shared"]) for snapshot in snapshots]
    B_private_list = [to_numpy(snapshot["B_private"]) for snapshot in snapshots]
    task_ids = [int(snapshot["task_id"]) for snapshot in snapshots]

    shared_norms = [float(np.linalg.norm(B)) for B in B_shared_list]
    private_norms = [float(np.linalg.norm(B)) for B in B_private_list]
    total_norms = [float((s ** 2 + p ** 2) ** 0.5) for s, p in zip(shared_norms, private_norms)]
    ratio = [float(s / (p + 1e-8)) for s, p in zip(shared_norms, private_norms)]

    metrics = {
        "task_ids": task_ids,
        "fro_norms_shared": shared_norms,
        "fro_norms_private": private_norms,
        "fro_norms_total": total_norms,
        "shared_private_ratio": ratio,
    }

    plot_line(
        task_ids,
        [shared_norms, private_norms, total_norms],
        ["shared", "private", "total"],
        "Frobenius norm",
        "Shared/Private Frobenius Norm by Task",
        os.path.join(output_dir, "shared_private_fro_norm_curve.png"),
    )
    plot_line(
        task_ids,
        [ratio],
        ["shared/private"],
        "Norm ratio",
        "Shared/Private Norm Ratio",
        os.path.join(output_dir, "shared_private_ratio_curve.png"),
    )

    plot_heatmap(
        padded_col_norm_matrix(B_shared_list),
        "B_shared Column Norm Heatmap",
        os.path.join(output_dir, "b_shared_col_norm_heatmap.png"),
    )
    plot_heatmap(
        padded_col_norm_matrix(B_private_list),
        "B_private Column Norm Heatmap",
        os.path.join(output_dir, "b_private_col_norm_heatmap.png"),
    )

    for task_id, B_shared, B_private in zip(task_ids, B_shared_list, B_private_list):
        plot_param_heatmap(
            B_shared,
            f"B_shared Heatmap Task {task_id}",
            os.path.join(output_dir, f"task_{task_id:02d}_B_shared_heatmap.png"),
        )
        plot_param_heatmap(
            B_private,
            f"B_private Heatmap Task {task_id}",
            os.path.join(output_dir, f"task_{task_id:02d}_B_private_heatmap.png"),
        )

    if with_svd:
        plot_svd_spectra(B_shared_list, "B_shared Singular Value Spectra", os.path.join(output_dir, "b_shared_svd_spectra.png"))
        plot_svd_spectra(B_private_list, "B_private Singular Value Spectra", os.path.join(output_dir, "b_private_svd_spectra.png"))

    save_metrics_json(metrics, os.path.join(output_dir, "b_shared_private_metrics.json"))


def main():
    parser = argparse.ArgumentParser(description="Visualize B snapshots saved from BOFA runs.")
    parser.add_argument("input_dir", type=str, help="Directory containing task_*_b_snapshot.pt files.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save plots. Defaults to <input_dir>/plots.")
    parser.add_argument("--no_svd", action="store_true", help="Disable singular value spectrum plots.")
    args = parser.parse_args()
    # '/home/shengqin/lys/code/BOFA/logs/bofa/imagenetr/0/20/seed_1993_model_bofa_20260505_102348/b_analysis/plots'
    output_dir = args.output_dir or os.path.join(args.input_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    # dict_keys(['task_id', 'subspace_policy', 'basis_alloc', 'active_rank', 'active_slice', 'B', 'slice'])
    snapshots = load_snapshots(args.input_dir)
    if "B" in snapshots[0]: #
        analyze_single_b(snapshots, output_dir, with_svd=not args.no_svd)
    else:
        analyze_shared_core(snapshots, output_dir, with_svd=not args.no_svd)

    print(f"Saved B analysis plots to {output_dir}")


if __name__ == "__main__":
    main()
