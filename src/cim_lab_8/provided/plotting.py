"""Figures for the lab. Matplotlib accepts CPU torch tensors directly.

Colours used everywhere: black = truth, blue = filter, orange = filter's uncertainty,
green = measurement used, red = measurement rejected by the gate.
"""
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch


def ellipse_points(mean, covariance, scale, n=60):
    """Points on the ellipse `scale` standard deviations out: mean + scale * L @ unit circle."""
    angle = torch.linspace(0, 2 * math.pi, n)
    circle = torch.stack([torch.cos(angle), torch.sin(angle)])
    L = torch.linalg.cholesky(covariance)  # a "square root" of the covariance
    return mean[:, None] + scale * L @ circle


def plot_trajectory(data, result, path, cfg, title):
    estimate = result.means[:, :2]
    z = data.measurements
    used = result.accepted
    rejected = data.observed & ~result.accepted
    fig, ax = plt.subplots(figsize=(8, 6))
    # The gap: thick grey band under the true path where there were no measurements.
    in_gap = data.true_observation_positions.clone()
    in_gap[data.observed] = torch.nan
    ax.plot(in_gap[:, 0], in_gap[:, 1], color="0.75", lw=10, solid_capstyle="round", label="Gap (no measurements)")
    ax.plot(data.fine_positions[:, 0], data.fine_positions[:, 1], color="black", lw=2, label="Truth")
    # Time labels every 2 seconds along the true path.
    for i in range(0, len(data.fine_times), round(2.0 / (data.fine_times[1] - data.fine_times[0]).item())):
        x, y = data.fine_positions[i]
        ax.plot(x, y, "o", color="black", ms=4)
        ax.annotate(f"{data.fine_times[i]:.0f} s", (x, y), xytext=(-24, 4), textcoords="offset points", fontsize=8)
    ax.scatter(z[used, 0], z[used, 1], s=30, marker="x", color="tab:green", label="Measurement (used)")
    ax.scatter(z[rejected, 0], z[rejected, 1], s=40, marker="x", color="tab:red", label="Measurement (rejected)")
    ax.scatter(z[data.is_outlier, 0], z[data.is_outlier, 1], s=150, facecolors="none", edgecolors="black",
               label="Deliberately wrong measurement")
    ax.plot(estimate[:, 0], estimate[:, 1], color="tab:blue", lw=2, label="Kalman filter")
    if result.restarted.any():
        ax.scatter(estimate[result.restarted, 0], estimate[result.restarted, 1], s=200, marker="*",
                   color="tab:purple", zorder=3, label="Track restarted")
    for t in range(0, len(estimate), cfg.plots.ellipse_stride):
        points = ellipse_points(estimate[t], result.covariances[t, :2, :2], cfg.plots.uncertainty_scale)
        ax.plot(points[0], points[1], color="tab:orange", lw=0.8, alpha=0.7,
                label="Filter's uncertainty (95% region)" if t == 0 else None)
    ax.set(title=title, xlabel="x position", ylabel="y position")
    ax.axis("equal"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=cfg.plots.dpi); plt.close(fig)


def plot_error(data, result, path, cfg):
    t = data.observation_times
    error = (result.means[:, :2] - data.true_observation_positions).norm(dim=1)
    sensor_error = (data.measurements - data.true_observation_positions).norm(dim=1)  # NaN in the gap
    # How far off the filter thinks it might be: the longest radius of its orange ellipse.
    uncertainty = cfg.plots.uncertainty_scale * torch.linalg.eigvalsh(result.covariances[:, :2, :2])[:, -1].sqrt()
    rejected = data.observed & ~result.accepted
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.scatter(t, sensor_error, s=12, color="0.6", label="Sensor error (measurement alone)")
    ax.scatter(t[data.is_outlier], sensor_error[data.is_outlier], s=120, facecolors="none", edgecolors="black",
               label="Deliberately wrong measurement")
    ax.plot(t, error, color="tab:blue", label="Filter error")
    ax.plot(t, uncertainty, color="tab:orange", label="Filter's uncertainty (error should be below this 95% of the time)")
    top = max(error.max(), uncertainty.max(), sensor_error.nan_to_num().max()).item() * 1.05
    ax.fill_between(t, 0, top, where=~data.observed, color="0.8", alpha=0.5, label="Gap (no measurements)")
    ax.plot(t[rejected], torch.zeros(int(rejected.sum())), "|", color="tab:red", ms=20, label="Rejected by gate")
    ax.set(xlabel="Time (s)", ylabel="Distance from the truth", title="Error and uncertainty over time")
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout(); fig.savefig(path, dpi=cfg.plots.dpi); plt.close(fig)


def plot_sweep(q_values, r_values, tables, current, path, cfg):
    """One heatmap per entry of `tables` ({title: (len(q), len(r)) tensor}).

    The cell matching `current` = (process_noise_std, observation_noise_std) is outlined.
    """
    fig, axes = plt.subplots(1, len(tables), figsize=(6 * len(tables), 4.5))
    for ax, (title, table) in zip(axes, tables.items()):
        image = ax.imshow(table, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(r_values)), r_values)
        ax.set_yticks(range(len(q_values)), q_values)
        ax.set(xlabel="observation_noise_std (trust in sensor: lower = more)",
               ylabel="process_noise_std (expected wandering)", title=title)
        middle = (table.min() + table.max()) / 2
        for i in range(len(q_values)):
            for j in range(len(r_values)):
                ax.text(j, i, f"{table[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if table[i, j] > middle else "white")
        if current[0] in q_values and current[1] in r_values:
            i, j = q_values.index(current[0]), r_values.index(current[1])
            for colour, width in [("black", 5), ("white", 2)]:  # visible on any background
                ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, edgecolor=colour, lw=width))
        fig.colorbar(image, ax=ax, label="lower is better")
    fig.suptitle("Each cell is a full run of the filter. Outlined: the settings you ran with")
    fig.tight_layout(); fig.savefig(path, dpi=cfg.plots.dpi); plt.close(fig)
