"""Run one experiment: simulate, filter, measure, plot, and sweep the noise settings."""
import csv
import json
from pathlib import Path
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from ..metrics import mean_position_error, position_rmse
from ..my_plots import plot_error_histogram
from .plotting import plot_error, plot_sweep, plot_trajectory
from .simulate import simulate_trajectory
from .tracker import run_filter

CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "conf")  # the repository's conf/ folder

# Output folders are named after the settings changed on the command line ("default" if none).
OmegaConf.register_new_resolver("run_name", lambda changed: changed or "default", replace=True)


def evaluate(data, result):
    """Numbers describing how well one filter run did."""
    truth = data.true_observation_positions
    return {
        "rmse": position_rmse(result.means, truth).item(),
        "rmse_when_measured": position_rmse(result.means, truth, data.observed).item(),
        "mean_error": mean_position_error(result.means, truth).item(),
        "gap_rmse": position_rmse(result.means, truth, ~data.observed).item(),
        "outliers_used": int((result.accepted & data.is_outlier).sum()),
        "good_measurements_rejected": int((data.observed & ~result.accepted & ~data.is_outlier).sum()),
        "restarts": int(result.restarted.sum()),
    }


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):
    output = Path.cwd()  # Hydra has already moved into this run's own outputs/ folder
    OmegaConf.save(cfg, output / "resolved_config.yaml")
    data = simulate_trajectory(cfg)
    truth = data.true_observation_positions

    # 1. Baseline: the sensor on its own, on the frames where it reports.
    summary = {"sensor_alone": {
        "rmse_when_measured": position_rmse(data.measurements, truth, data.observed).item(),
        "mean_error": mean_position_error(data.measurements, truth, data.observed).item(),
    }}

    # 2. The Kalman filter with the configured settings.
    result = run_filter(data, cfg)
    summary["filter"] = evaluate(data, result)

    title = cfg.filter.model.replace("_", " ").capitalize() + " model"
    plot_trajectory(data, result, output / "kalman_trajectory.png", cfg, title)
    plot_error(data, result, output / "uncertainty_and_error.png", cfg)
    try:
        sensor_errors = (data.measurements - truth).norm(dim=1)
        filter_errors = (result.means[:, :2] - truth).norm(dim=1)
        plot_error_histogram(sensor_errors, filter_errors, output / "error_histogram.png")
    except NotImplementedError:
        print("Task 3 not done yet: skipping error_histogram.png")

    # 3. Sweep: rerun the filter for every combination of assumed noise levels.
    q_values = list(cfg.experiment.process_noise_values)
    r_values = list(cfg.experiment.observation_noise_values)
    rmse = torch.zeros(len(q_values), len(r_values))
    gap_rmse = torch.zeros(len(q_values), len(r_values))
    rows = []
    for i, q in enumerate(q_values):
        for j, r in enumerate(r_values):
            metrics = evaluate(data, run_filter(data, cfg, process_noise_std=q, observation_noise_std=r))
            rmse[i, j], gap_rmse[i, j] = metrics["rmse"], metrics["gap_rmse"]
            rows.append({"process_noise_std": q, "observation_noise_std": r, **metrics})
    with open(output / "noise_sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    current = (cfg.filter.process_noise_std, cfg.filter.observation_noise_std)
    plot_sweep(q_values, r_values, {"Error (RMSE), all frames": rmse, "Error (RMSE) during the gap": gap_rmse},
               current, output / "noise_sweep.png", cfg)

    # 4. Optional: let PyTorch learn the noise levels by gradient descent.
    if cfg.experiment.fit_noise:
        from .learning import fit_noise
        q, r = fit_noise(data, cfg)
        learned = run_filter(data, cfg, process_noise_std=q, observation_noise_std=r)
        summary["learned_noise"] = {"process_noise_std": q, "observation_noise_std": r, **evaluate(data, learned)}

    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nFigures saved in {output}")


if __name__ == "__main__":
    main()
