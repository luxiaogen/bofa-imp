import argparse
import ast
import glob
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_snapshots(input_dir):
    snapshot_paths = sorted(glob.glob(os.path.join(input_dir, "task_*_b_snapshot.pt")))
    if not snapshot_paths:
        raise FileNotFoundError(f"No task_*_b_snapshot.pt found under {input_dir}")
    return [torch.load(path, map_location="cpu") for path in snapshot_paths]


def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def find_log_for_analysis_dir(input_dir):
    analysis_path = Path(input_dir).resolve()
    run_dir = analysis_path.parent
    candidate = run_dir.with_suffix(".log")
    if candidate.exists():
        return str(candidate)
    matches = sorted(glob.glob(str(run_dir.parent / f"{run_dir.name}*.log")))
    return matches[0] if matches else None


def parse_curve(label, text):
    matches = re.findall(rf"{re.escape(label)}: (\[[^\n]+\])", text)
    if not matches:
        return []
    try:
        return [float(x) for x in ast.literal_eval(matches[-1])]
    except (SyntaxError, ValueError):
        return []


def parse_all_scalars(label, text):
    matches = re.findall(rf"{re.escape(label)}: ([0-9.]+)", text)
    return [float(x) for x in matches]


def parse_log_metrics(input_dir):
    log_path = find_log_for_analysis_dir(input_dir)
    if not log_path:
        return {}
    text = read_text(log_path)
    return {
        "log_path": log_path,
        "top1_curve": parse_curve("CNN top1 curve", text),
        "top2_curve": parse_curve("CNN top2 curve", text),
        "task_top1_curve": parse_all_scalars("Average Task Accuracy (CNN)", text),
        "task_top2_curve": parse_all_scalars("Average Task Accuracy (CNN Top2)", text),
    }


def padded_col_norm_matrix(arrays):
    max_cols = max(arr.shape[1] for arr in arrays)
    heatmap = np.full((len(arrays), max_cols), np.nan, dtype=np.float32)
    for i, arr in enumerate(arrays):
        norms = np.linalg.norm(arr, axis=0)
        heatmap[i, : len(norms)] = norms
    return heatmap


def singular_values(matrix):
    return np.linalg.svd(matrix, compute_uv=False)


def effective_rank_from_singular_values(values, eps=1e-12):
    energy = np.square(values)
    total = float(energy.sum())
    if total <= eps:
        return 0.0
    probs = energy / total
    entropy = -float(np.sum(probs * np.log(probs + eps)))
    return float(np.exp(entropy))


def topk_energy_ratio(values, topk=20, eps=1e-12):
    energy = np.square(values)
    total = float(energy.sum())
    if total <= eps:
        return 0.0
    return float(energy[: min(topk, len(energy))].sum() / total)


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


def plot_accuracy_and_delta(task_ids, log_metrics, delta_series, output_path):
    has_acc = bool(log_metrics.get("top1_curve") or log_metrics.get("top2_curve") or log_metrics.get("task_top1_curve"))
    if not has_acc or not delta_series:
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    acc_ax = axes[0]
    for key, label in [
        ("top1_curve", "CNN top1"),
        ("top2_curve", "CNN top2"),
        ("task_top1_curve", "task top1"),
    ]:
        curve = log_metrics.get(key) or []
        if curve:
            acc_ax.plot(np.arange(1, len(curve) + 1), curve, marker="o", label=label)
    acc_ax.set_ylabel("Accuracy")
    acc_ax.set_title("Accuracy and B Update Delta")
    acc_ax.grid(alpha=0.25)
    acc_ax.legend()

    delta_ax = axes[1]
    for values, label in delta_series:
        delta_ax.plot(task_ids, values, marker="o", label=label)
    delta_ax.set_xlabel("Task")
    delta_ax.set_ylabel("Frobenius delta")
    delta_ax.grid(alpha=0.25)
    delta_ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_update_ratio(task_ids, shared_deltas, private_deltas, output_path):
    denom = np.asarray(shared_deltas) + np.asarray(private_deltas) + 1e-8
    shared_ratio = np.asarray(shared_deltas) / denom
    private_ratio = np.asarray(private_deltas) / denom

    plt.figure(figsize=(9, 4))
    x = np.arange(len(task_ids))
    plt.bar(x, shared_ratio, label="shared update")
    plt.bar(x, private_ratio, bottom=shared_ratio, label="private update")
    plt.xticks(x, task_ids)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Task")
    plt.ylabel("Update ratio")
    plt.title("Shared vs Private Update Ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_effective_rank(task_ids, arrays_by_name, output_path):
    plt.figure(figsize=(8, 4))
    for name, arrays in arrays_by_name:
        values = [effective_rank_from_singular_values(singular_values(arr)) for arr in arrays]
        plt.plot(task_ids, values, marker="o", label=name)
    plt.xlabel("Task")
    plt.ylabel("Effective rank")
    plt.title("Effective Rank by Task")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_topk_energy_ratio(task_ids, arrays_by_name, output_path, topk=20):
    plt.figure(figsize=(8, 4))
    for name, arrays in arrays_by_name:
        values = [topk_energy_ratio(singular_values(arr), topk=topk) for arr in arrays]
        plt.plot(task_ids, values, marker="o", label=f"{name} top{topk}")
    plt.xlabel("Task")
    plt.ylabel("Energy ratio")
    plt.ylim(0.0, 1.02)
    plt.title(f"Top-{topk} Singular Energy Ratio")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_importance_vs_delta(snapshots, B_shared_list, output_path):
    xs = []
    ys = []
    colors = []
    for i in range(1, len(snapshots)):
        prev = snapshots[i - 1]
        if "shared_importance" in prev:
            importance = to_numpy(prev["shared_importance"]).reshape(-1)
            if importance.size == B_shared_list[i].shape[1] and np.nanmax(importance) > 0:
                delta_cols = np.linalg.norm(B_shared_list[i] - B_shared_list[i - 1], axis=0)
                xs.extend(importance.tolist())
                ys.extend(delta_cols.tolist())
                colors.extend([int(snapshots[i]["task_id"])] * len(delta_cols))
        if "shared_param_importance" in prev:
            importance = to_numpy(prev["shared_param_importance"])
            if importance.shape == B_shared_list[i].shape and np.nanmax(importance) > 0:
                col_importance = importance.mean(axis=0)
                delta_cols = np.linalg.norm(B_shared_list[i] - B_shared_list[i - 1], axis=0)
                xs.extend(col_importance.tolist())
                ys.extend(delta_cols.tolist())
                colors.extend([int(snapshots[i]["task_id"])] * len(delta_cols))

    if not xs:
        return

    corr = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 and np.std(xs) > 0 and np.std(ys) > 0 else 0.0
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(xs, ys, c=colors, s=12, alpha=0.55, cmap="viridis")
    plt.colorbar(scatter, label="Task")
    plt.xlabel("Previous importance")
    plt.ylabel("Current column delta norm")
    plt.title(f"Importance vs Delta (corr={corr:.3f})")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_svd_anchor_weights(snapshots, task_ids, output_dir):
    weights = []
    used_task_ids = []
    for snapshot in snapshots:
        if "shared_svd_weight" not in snapshot:
            continue
        weight = to_numpy(snapshot["shared_svd_weight"]).astype(np.float32).reshape(-1)
        if weight.size == 0 or np.nanmax(np.abs(weight)) <= 0:
            continue
        weights.append(weight)
        used_task_ids.append(int(snapshot["task_id"]))
    if not weights:
        return {}

    max_len = max(weight.size for weight in weights)
    matrix = np.full((len(weights), max_len), np.nan, dtype=np.float32)
    for i, weight in enumerate(weights):
        matrix[i, : weight.size] = weight

    plot_scale_heatmap(
        matrix,
        "SVD Anchor Weight Heatmap",
        os.path.join(output_dir, "method_svd_anchor_weight_heatmap.png"),
        task_ids=used_task_ids,
        center=1.0,
    )
    weight_min = [float(np.nanmin(weight)) for weight in weights]
    weight_mean = [float(np.nanmean(weight)) for weight in weights]
    weight_max = [float(np.nanmax(weight)) for weight in weights]
    plot_line(
        used_task_ids,
        [weight_min, weight_mean, weight_max],
        ["min", "mean", "max"],
        "SVD weight",
        "SVD Anchor Weight by Task",
        os.path.join(output_dir, "method_svd_anchor_weight_curve.png"),
    )
    return {
        "svd_anchor_weight_tasks": used_task_ids,
        "svd_anchor_weight_min": weight_min,
        "svd_anchor_weight_mean": weight_mean,
        "svd_anchor_weight_max": weight_max,
    }


def plot_protected_subspace_drift(snapshots, B_shared_list, output_dir):
    transition_tasks = []
    protected_norms = []
    residual_norms = []
    protected_ratios = []
    weighted_projected_norms = []

    for i in range(1, len(B_shared_list)):
        prev_B = B_shared_list[i - 1]
        cur_B = B_shared_list[i]
        delta = cur_B - prev_B
        total_norm = float(np.linalg.norm(delta))
        if total_norm <= 1e-12:
            continue

        topk = int(snapshots[i - 1].get("shared_svd_reg_topk", 20))
        topk = max(1, min(topk, prev_B.shape[0], prev_B.shape[1]))
        U, S, Vh = np.linalg.svd(prev_B, full_matrices=False)
        U_k = U[:, :topk]
        V_k = Vh[:topk, :].T
        projected = U_k.T @ delta @ V_k
        protected_norm = float(np.linalg.norm(projected))
        residual_norm = float(max(total_norm ** 2 - protected_norm ** 2, 0.0) ** 0.5)
        protected_ratio = float((protected_norm ** 2) / (total_norm ** 2 + 1e-12))

        weight = None
        if "shared_svd_weight" in snapshots[i - 1]:
            weight = to_numpy(snapshots[i - 1]["shared_svd_weight"]).reshape(-1)[:topk]
        if weight is not None and weight.size == topk and np.nanmax(np.abs(weight)) > 0:
            weight_matrix = np.sqrt(np.outer(weight, weight))
            weighted_projected_norm = float(np.linalg.norm(projected * weight_matrix))
        else:
            weighted_projected_norm = protected_norm

        transition_tasks.append(int(snapshots[i]["task_id"]))
        protected_norms.append(protected_norm)
        residual_norms.append(residual_norm)
        protected_ratios.append(protected_ratio)
        weighted_projected_norms.append(weighted_projected_norm)

    if not transition_tasks:
        return {}

    plot_line(
        transition_tasks,
        [protected_norms, residual_norms, weighted_projected_norms],
        ["protected top-k delta", "orthogonal residual", "weighted protected delta"],
        "Delta norm",
        "SVD/OGD Protected Subspace Drift",
        os.path.join(output_dir, "method_svd_ogd_protected_drift_curve.png"),
    )
    plot_line(
        transition_tasks,
        [protected_ratios],
        ["protected energy ratio"],
        "Energy ratio",
        "SVD/OGD Protected Subspace Delta Ratio",
        os.path.join(output_dir, "method_svd_ogd_protected_ratio_curve.png"),
    )
    return {
        "protected_transition_tasks": transition_tasks,
        "protected_subspace_delta_norm": protected_norms,
        "orthogonal_residual_delta_norm": residual_norms,
        "weighted_protected_subspace_delta_norm": weighted_projected_norms,
        "protected_subspace_delta_ratio": protected_ratios,
    }


def plot_ewc_importance_diagnostics(snapshots, B_shared_list, output_dir):
    task_ids = []
    col_importance_rows = []
    transition_tasks = []
    ewc_penalty = []
    high_importance_delta = []
    low_importance_delta = []
    high_low_ratio = []
    param_corrs = []
    col_corrs = []

    for i, snapshot in enumerate(snapshots):
        if "shared_param_importance" not in snapshot:
            continue
        importance = to_numpy(snapshot["shared_param_importance"]).astype(np.float32)
        if importance.ndim != 2 or importance.shape != B_shared_list[i].shape or np.nanmax(importance) <= 0:
            continue
        task_ids.append(int(snapshot["task_id"]))
        col_importance_rows.append(np.mean(importance, axis=0))

    if col_importance_rows:
        matrix = np.stack(col_importance_rows, axis=0)
        plot_scale_heatmap(
            matrix,
            "EWC Column Importance Heatmap",
            os.path.join(output_dir, "method_ewc_column_importance_heatmap.png"),
            task_ids=task_ids,
            center=1.0,
        )
        plot_line(
            task_ids,
            [
                [float(np.min(row)) for row in col_importance_rows],
                [float(np.mean(row)) for row in col_importance_rows],
                [float(np.max(row)) for row in col_importance_rows],
            ],
            ["min", "mean", "max"],
            "Column importance",
            "EWC Column Importance by Task",
            os.path.join(output_dir, "method_ewc_column_importance_curve.png"),
        )

    for i in range(1, len(snapshots)):
        prev = snapshots[i - 1]
        if "shared_param_importance" not in prev:
            continue
        importance = to_numpy(prev["shared_param_importance"]).astype(np.float32)
        if importance.ndim != 2 or importance.shape != B_shared_list[i].shape or np.nanmax(importance) <= 0:
            continue
        delta = B_shared_list[i] - B_shared_list[i - 1]
        abs_delta = np.abs(delta)
        flat_importance = importance.reshape(-1)
        flat_delta = abs_delta.reshape(-1)
        p20 = np.percentile(flat_importance, 20)
        p80 = np.percentile(flat_importance, 80)
        high_mask = flat_importance >= p80
        low_mask = flat_importance <= p20
        high_delta = float(np.mean(flat_delta[high_mask])) if np.any(high_mask) else 0.0
        low_delta = float(np.mean(flat_delta[low_mask])) if np.any(low_mask) else 0.0
        col_importance = np.mean(importance, axis=0)
        col_delta = np.linalg.norm(delta, axis=0)
        param_corr = (
            float(np.corrcoef(flat_importance, flat_delta)[0, 1])
            if np.std(flat_importance) > 0 and np.std(flat_delta) > 0
            else 0.0
        )
        col_corr = (
            float(np.corrcoef(col_importance, col_delta)[0, 1])
            if np.std(col_importance) > 0 and np.std(col_delta) > 0
            else 0.0
        )

        transition_tasks.append(int(snapshots[i]["task_id"]))
        ewc_penalty.append(float(np.sum(importance * np.square(delta))))
        high_importance_delta.append(high_delta)
        low_importance_delta.append(low_delta)
        high_low_ratio.append(float(high_delta / (low_delta + 1e-12)))
        param_corrs.append(param_corr)
        col_corrs.append(col_corr)

    if transition_tasks:
        plot_line(
            transition_tasks,
            [ewc_penalty],
            ["EWC weighted delta"],
            "sum(I * delta^2)",
            "EWC Weighted Drift by Task",
            os.path.join(output_dir, "method_ewc_weighted_drift_curve.png"),
        )
        plot_line(
            transition_tasks,
            [high_importance_delta, low_importance_delta],
            ["top 20% importance", "bottom 20% importance"],
            "Mean abs delta",
            "EWC Important vs Unimportant Parameter Drift",
            os.path.join(output_dir, "method_ewc_important_vs_unimportant_delta_curve.png"),
        )
        plot_line(
            transition_tasks,
            [high_low_ratio],
            ["top/bottom delta ratio"],
            "Ratio",
            "EWC Important Parameter Drift Ratio",
            os.path.join(output_dir, "method_ewc_important_delta_ratio_curve.png"),
        )
        plot_line(
            transition_tasks,
            [param_corrs, col_corrs],
            ["parameter corr", "column corr"],
            "Correlation",
            "EWC Importance Delta Correlation",
            os.path.join(output_dir, "method_ewc_importance_delta_corr_curve.png"),
        )

    return {
        "ewc_importance_tasks": task_ids,
        "ewc_transition_tasks": transition_tasks,
        "ewc_weighted_delta": ewc_penalty,
        "ewc_high_importance_delta": high_importance_delta,
        "ewc_low_importance_delta": low_importance_delta,
        "ewc_high_low_delta_ratio": high_low_ratio,
        "ewc_param_importance_delta_corr": param_corrs,
        "ewc_col_importance_delta_corr": col_corrs,
    }


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


def plot_scale_heatmap(matrix, title, output_path, task_ids=None, center=None):
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return
    vmin = float(np.percentile(finite, 1))
    vmax = float(np.percentile(finite, 99))
    if abs(vmax - vmin) < 1e-8:
        vmin -= 1e-4
        vmax += 1e-4
    plt.figure(figsize=(10, 4))
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color="lightgray")
    if center is not None and vmin < center < vmax:
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        plt.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    else:
        plt.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlabel("Shared direction")
    plt.ylabel("Task")
    if task_ids is not None:
        plt.yticks(np.arange(len(task_ids)), task_ids)
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
    plt.figure(figsize=(8, 4))
    max_plot_len = 0
    for task_id, arr in enumerate(arrays):
        singular_values = np.linalg.svd(arr, compute_uv=False)[:topk]
        max_plot_len = max(max_plot_len, len(singular_values))
        plt.plot(np.arange(1, len(singular_values) + 1), singular_values, marker="o", label=f"task_{task_id}")
    plt.xlabel("Singular value index")
    plt.ylabel("Singular value")
    if max_plot_len > 0:
        plt.xticks(np.arange(1, max_plot_len + 1, 2))
    plt.title(title)
    if len(arrays) <= 10:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_singular_value_diagnostics(snapshots, B_shared_list, output_dir):
    if not B_shared_list:
        return {}

    task_ids = [int(snapshot["task_id"]) for snapshot in snapshots]
    configured_topk = max(int(snapshot.get("shared_svd_reg_topk", 20)) for snapshot in snapshots)
    plot_topk = max(1, min(configured_topk, 50, B_shared_list[0].shape[0], B_shared_list[0].shape[1]))
    singular_matrix = np.stack([singular_values(B)[:plot_topk] for B in B_shared_list], axis=0)

    plot_heatmap(
        singular_matrix,
        "B_shared Top Singular Values",
        os.path.join(output_dir, "method_svd_shared_singular_values_heatmap.png"),
        xlabel="Singular value index",
        ylabel="Task",
    )

    top1 = singular_matrix[:, 0].tolist()
    top5_mean = singular_matrix[:, : min(5, plot_topk)].mean(axis=1).tolist()
    top20_mean = singular_matrix[:, : min(20, plot_topk)].mean(axis=1).tolist()
    plot_line(
        task_ids,
        [top1, top5_mean, top20_mean],
        ["sigma1", "mean top5", "mean top20"],
        "Singular value",
        "B_shared Top Singular Values by Task",
        os.path.join(output_dir, "method_svd_shared_top_singular_values_curve.png"),
    )

    if len(B_shared_list) < 2:
        return {
            "svd_singular_value_tasks": task_ids,
            "svd_sigma1": top1,
            "svd_top5_mean": top5_mean,
            "svd_top20_mean": top20_mean,
        }

    delta_matrix = singular_matrix[1:] - singular_matrix[:-1]
    transition_tasks = task_ids[1:]
    plot_scale_heatmap(
        delta_matrix,
        "B_shared Singular Value Change",
        os.path.join(output_dir, "method_svd_shared_singular_value_delta_heatmap.png"),
        task_ids=transition_tasks,
        center=0.0,
    )

    topk = max(1, min(configured_topk, B_shared_list[0].shape[0], B_shared_list[0].shape[1]))
    top_delta_norms = []
    tail_delta_norms = []
    top_delta_abs_mean = []
    projected_diag_rows = []
    projected_offdiag_norms = []
    projected_diag_abs_mean = []

    for i in range(1, len(B_shared_list)):
        prev_s = singular_values(B_shared_list[i - 1])
        cur_s = singular_values(B_shared_list[i])
        top_len = min(topk, len(prev_s), len(cur_s))
        top_delta = cur_s[:top_len] - prev_s[:top_len]
        tail_delta = cur_s[top_len:] - prev_s[top_len:]
        top_delta_norms.append(float(np.linalg.norm(top_delta)))
        tail_delta_norms.append(float(np.linalg.norm(tail_delta)))
        top_delta_abs_mean.append(float(np.mean(np.abs(top_delta))) if top_delta.size > 0 else 0.0)

        prev_B = B_shared_list[i - 1]
        delta_B = B_shared_list[i] - prev_B
        U, _, Vh = np.linalg.svd(prev_B, full_matrices=False)
        diag_len = min(plot_topk, top_len)
        U_k = U[:, :diag_len]
        V_k = Vh[:diag_len, :].T
        projected = U_k.T @ delta_B @ V_k
        projected_diag = np.diag(projected)
        projected_diag_rows.append(projected_diag)
        offdiag = projected - np.diag(projected_diag)
        projected_offdiag_norms.append(float(np.linalg.norm(offdiag)))
        projected_diag_abs_mean.append(float(np.mean(np.abs(projected_diag))) if projected_diag.size > 0 else 0.0)

    plot_line(
        transition_tasks,
        [top_delta_norms, tail_delta_norms, top_delta_abs_mean],
        ["top-k sigma delta norm", "tail sigma delta norm", "mean abs top-k sigma delta"],
        "Singular value delta",
        "B_shared Singular Value Drift by Task",
        os.path.join(output_dir, "method_svd_shared_top_tail_singular_delta_curve.png"),
    )

    projected_diag_matrix = np.stack(projected_diag_rows, axis=0)
    plot_scale_heatmap(
        projected_diag_matrix,
        "Projected Delta on Old SVD Diagonal Directions",
        os.path.join(output_dir, "method_svd_projected_diag_delta_heatmap.png"),
        task_ids=transition_tasks,
        center=0.0,
    )
    plot_line(
        transition_tasks,
        [projected_diag_abs_mean, projected_offdiag_norms],
        ["mean abs protected diagonal delta", "protected off-diagonal delta norm"],
        "Projected delta",
        "SVD Protected Coordinate Drift",
        os.path.join(output_dir, "method_svd_projected_coordinate_drift_curve.png"),
    )

    return {
        "svd_singular_value_tasks": task_ids,
        "svd_transition_tasks": transition_tasks,
        "svd_sigma1": top1,
        "svd_top5_mean": top5_mean,
        "svd_top20_mean": top20_mean,
        "svd_topk_sigma_delta_norm": top_delta_norms,
        "svd_tail_sigma_delta_norm": tail_delta_norms,
        "svd_topk_sigma_delta_abs_mean": top_delta_abs_mean,
        "svd_projected_diag_abs_mean": projected_diag_abs_mean,
        "svd_projected_offdiag_norm": projected_offdiag_norms,
    }


def plot_direct_singular_value_changes(snapshots, B_shared_list, output_dir):
    if len(B_shared_list) < 2:
        return {}

    task_ids = [int(snapshot["task_id"]) for snapshot in snapshots]
    configured_topk = max(int(snapshot.get("shared_svd_reg_topk", 20)) for snapshot in snapshots)
    full_rank = min(B_shared_list[0].shape[0], B_shared_list[0].shape[1])
    plot_topk = max(1, min(configured_topk, 50, full_rank))
    protected_topk = max(1, min(configured_topk, full_rank))

    all_singular_values = [singular_values(B) for B in B_shared_list]
    transition_tasks = task_ids[1:]
    relative_rows = []
    abs_relative_rows = []
    topk_abs_relative_mean = []
    tail_abs_relative_mean = []
    topk_abs_delta_mean = []
    tail_abs_delta_mean = []
    topk_changed_ratio_1pct = []
    topk_changed_ratio_5pct = []

    for i in range(1, len(all_singular_values)):
        prev_s = all_singular_values[i - 1]
        cur_s = all_singular_values[i]
        rel = (cur_s[:plot_topk] - prev_s[:plot_topk]) / (np.abs(prev_s[:plot_topk]) + 1e-12)
        relative_rows.append(rel)
        abs_relative_rows.append(np.abs(rel))

        top_rel = (cur_s[:protected_topk] - prev_s[:protected_topk]) / (np.abs(prev_s[:protected_topk]) + 1e-12)
        top_abs_delta = np.abs(cur_s[:protected_topk] - prev_s[:protected_topk])
        tail_rel = np.array([], dtype=np.float32)
        tail_abs_delta = np.array([], dtype=np.float32)
        if protected_topk < len(prev_s):
            tail_rel = (cur_s[protected_topk:] - prev_s[protected_topk:]) / (
                np.abs(prev_s[protected_topk:]) + 1e-12
            )
            tail_abs_delta = np.abs(cur_s[protected_topk:] - prev_s[protected_topk:])

        topk_abs_relative_mean.append(float(np.mean(np.abs(top_rel))))
        tail_abs_relative_mean.append(float(np.mean(np.abs(tail_rel))) if tail_rel.size > 0 else 0.0)
        topk_abs_delta_mean.append(float(np.mean(top_abs_delta)))
        tail_abs_delta_mean.append(float(np.mean(tail_abs_delta)) if tail_abs_delta.size > 0 else 0.0)
        topk_changed_ratio_1pct.append(float(np.mean(np.abs(top_rel) > 0.01)))
        topk_changed_ratio_5pct.append(float(np.mean(np.abs(top_rel) > 0.05)))

    relative_matrix = np.stack(relative_rows, axis=0)
    abs_relative_matrix = np.stack(abs_relative_rows, axis=0)

    plot_scale_heatmap(
        relative_matrix * 100.0,
        "B_shared Singular Value Relative Change (%)",
        os.path.join(output_dir, "method_svd_singular_value_relative_change_heatmap.png"),
        task_ids=transition_tasks,
        center=0.0,
    )
    plot_heatmap(
        abs_relative_matrix * 100.0,
        "B_shared Singular Value Absolute Relative Change (%)",
        os.path.join(output_dir, "method_svd_singular_value_abs_relative_change_heatmap.png"),
        xlabel="Singular value index",
        ylabel="Task transition",
    )

    plot_line(
        transition_tasks,
        [topk_abs_relative_mean, tail_abs_relative_mean],
        ["protected top-k mean |relative change|", "tail mean |relative change|"],
        "Mean absolute relative change",
        "Direct Singular Value Change: Protected Top-k vs Tail",
        os.path.join(output_dir, "method_svd_direct_topk_tail_relative_change_curve.png"),
    )
    plot_line(
        transition_tasks,
        [topk_abs_delta_mean, tail_abs_delta_mean],
        ["protected top-k mean |delta sigma|", "tail mean |delta sigma|"],
        "Mean absolute singular value change",
        "Direct Singular Value Delta: Protected Top-k vs Tail",
        os.path.join(output_dir, "method_svd_direct_topk_tail_abs_delta_curve.png"),
    )
    plot_line(
        transition_tasks,
        [topk_changed_ratio_1pct, topk_changed_ratio_5pct],
        ["top-k changed > 1%", "top-k changed > 5%"],
        "Fraction of protected singular values",
        "Protected Singular Values Changed Ratio",
        os.path.join(output_dir, "method_svd_protected_changed_ratio_curve.png"),
    )

    prev_s = all_singular_values[-2][:plot_topk]
    cur_s = all_singular_values[-1][:plot_topk]
    x = np.arange(1, plot_topk + 1)
    plt.figure(figsize=(9, 4))
    plt.plot(x, prev_s, marker="o", label=f"task {task_ids[-2]}")
    plt.plot(x, cur_s, marker="o", label=f"task {task_ids[-1]}")
    for idx, before, after in zip(x, prev_s, cur_s):
        plt.plot([idx, idx], [before, after], color="gray", alpha=0.35, linewidth=1)
    plt.xlabel("Singular value index")
    plt.ylabel("Singular value")
    plt.title("Last Transition Singular Values: Before vs After")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "method_svd_last_transition_before_after.png"), dpi=200)
    plt.close()

    return {
        "svd_direct_transition_tasks": transition_tasks,
        "svd_topk_abs_relative_change_mean": topk_abs_relative_mean,
        "svd_tail_abs_relative_change_mean": tail_abs_relative_mean,
        "svd_topk_abs_delta_mean": topk_abs_delta_mean,
        "svd_tail_abs_delta_mean": tail_abs_delta_mean,
        "svd_topk_changed_ratio_1pct": topk_changed_ratio_1pct,
        "svd_topk_changed_ratio_5pct": topk_changed_ratio_5pct,
    }


def save_metrics_json(metrics, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def analyze_single_b(snapshots, output_dir, with_svd=True):
    Bs = [to_numpy(snapshot["B"]) for snapshot in snapshots]
    task_ids = [int(snapshot["task_id"]) for snapshot in snapshots]

    fro_norms = [float(np.linalg.norm(B)) for B in Bs]
    mean_abs = [float(np.abs(B).mean()) for B in Bs]
    max_abs = [float(np.abs(B).max()) for B in Bs]
    metrics = {
        "task_ids": task_ids,
        "fro_norms": fro_norms,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
    }

    plot_line(task_ids, [fro_norms], ["B"], "Frobenius norm", "B Frobenius Norm by Task",
              os.path.join(output_dir, "b_fro_norm_curve.png"))
    plot_heatmap(padded_col_norm_matrix(Bs), "B Column Norm Heatmap", os.path.join(output_dir, "b_col_norm_heatmap.png"))

    for task_id, B in zip(task_ids, Bs):
        plot_param_heatmap(B, f"B Heatmap Task {task_id}", os.path.join(output_dir, f"task_{task_id:02d}_B_heatmap.png"))

    if with_svd:
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
    shared_deltas = [shared_norms[0]]
    private_deltas = [private_norms[0]]
    for i in range(1, len(B_shared_list)):
        shared_deltas.append(float(np.linalg.norm(B_shared_list[i] - B_shared_list[i - 1])))
        private_deltas.append(float(np.linalg.norm(B_private_list[i] - B_private_list[i - 1])))

    metrics = {
        "task_ids": task_ids,
        "fro_norms_shared": shared_norms,
        "fro_norms_private": private_norms,
        "fro_norms_total": total_norms,
        "shared_private_ratio": ratio,
        "delta_fro_norm_shared": shared_deltas,
        "delta_fro_norm_private": private_deltas,
        "private_update_ratio": [
            float(p / (s + p + 1e-8)) for s, p in zip(shared_deltas, private_deltas)
        ],
    }

    grad_scale_list = []
    if all("shared_grad_scale" in snapshot for snapshot in snapshots):
        grad_scale_list = [to_numpy(snapshot["shared_grad_scale"]).astype(np.float32) for snapshot in snapshots]
        scale_matrix = np.stack(grad_scale_list, axis=0)
        scale_min = [float(np.min(scale)) for scale in grad_scale_list]
        scale_mean = [float(np.mean(scale)) for scale in grad_scale_list]
        scale_max = [float(np.max(scale)) for scale in grad_scale_list]
        metrics.update(
            {
                "shared_grad_scale_min": scale_min,
                "shared_grad_scale_mean": scale_mean,
                "shared_grad_scale_max": scale_max,
            }
        )
        plot_line(
            task_ids,
            [scale_min, scale_mean, scale_max],
            ["min", "mean", "max"],
            "Gradient scale",
            "B_shared Gradient Scale by Task",
            os.path.join(output_dir, "shared_grad_scale_curve.png"),
        )
        plot_scale_heatmap(
            scale_matrix,
            "B_shared Gradient Scale Heatmap",
            os.path.join(output_dir, "shared_grad_scale_heatmap.png"),
            task_ids=task_ids,
        )
        relative_scale_matrix = scale_matrix / (scale_matrix.mean(axis=1, keepdims=True) + 1e-8)
        plot_scale_heatmap(
            relative_scale_matrix,
            "B_shared Relative Gradient Scale Heatmap",
            os.path.join(output_dir, "shared_grad_scale_relative_heatmap.png"),
            task_ids=task_ids,
            center=1.0,
        )

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
    plot_line(
        task_ids,
        [shared_deltas, private_deltas],
        ["delta shared", "delta private"],
        "Frobenius delta",
        "Shared/Private Delta by Task",
        os.path.join(output_dir, "diagnostic_shared_private_delta_curve.png"),
    )
    plot_update_ratio(
        task_ids,
        shared_deltas,
        private_deltas,
        os.path.join(output_dir, "diagnostic_shared_private_update_ratio.png"),
    )
    plot_effective_rank(
        task_ids,
        [("B_shared", B_shared_list), ("B_private", B_private_list)],
        os.path.join(output_dir, "diagnostic_effective_rank_curve.png"),
    )
    plot_topk_energy_ratio(
        task_ids,
        [("B_shared", B_shared_list), ("B_private", B_private_list)],
        os.path.join(output_dir, "diagnostic_top20_energy_ratio_curve.png"),
        topk=20,
    )
    plot_importance_vs_delta(
        snapshots,
        B_shared_list,
        os.path.join(output_dir, "diagnostic_importance_vs_delta.png"),
    )
    metrics.update(plot_singular_value_diagnostics(snapshots, B_shared_list, output_dir))
    metrics.update(plot_direct_singular_value_changes(snapshots, B_shared_list, output_dir))
    metrics.update(plot_svd_anchor_weights(snapshots, task_ids, output_dir))
    metrics.update(plot_protected_subspace_drift(snapshots, B_shared_list, output_dir))
    metrics.update(plot_ewc_importance_diagnostics(snapshots, B_shared_list, output_dir))

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

    output_dir = args.output_dir or os.path.join(args.input_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    snapshots = load_snapshots(args.input_dir)
    if "B" in snapshots[0]:
        analyze_single_b(snapshots, output_dir, with_svd=not args.no_svd)
    else:
        analyze_shared_core(snapshots, output_dir, with_svd=not args.no_svd)
        log_metrics = parse_log_metrics(args.input_dir)
        metrics_path = os.path.join(output_dir, "b_shared_private_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            plot_accuracy_and_delta(
                metrics["task_ids"],
                log_metrics,
                [
                    (metrics["delta_fro_norm_shared"], "delta shared"),
                    (metrics["delta_fro_norm_private"], "delta private"),
                ],
                os.path.join(output_dir, "diagnostic_accuracy_and_delta_curve.png"),
            )
            if log_metrics:
                metrics["log_metrics"] = log_metrics
                save_metrics_json(metrics, metrics_path)

    print(f"Saved B analysis plots to {output_dir}")


if __name__ == "__main__":
    main()
