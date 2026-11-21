"""Simulate a smooth 2D trajectory with abrupt turns, and coarse, noisy, incomplete measurements."""
from dataclasses import dataclass
import torch


@dataclass
class TrajectoryData:
    fine_times: torch.Tensor                  # (N,) time grid of the true motion
    fine_positions: torch.Tensor              # (N, 2)
    observation_times: torch.Tensor           # (T,) frame times: when the sensor could report
    true_observation_positions: torch.Tensor  # (T, 2) true position at each frame
    measurements: torch.Tensor                # (T, 2) noisy positions, NaN where missing
    observed: torch.Tensor                    # (T,) False during a gap (no measurement)
    is_outlier: torch.Tensor                  # (T,) True for injected wrong measurements


def simulate_trajectory(cfg):
    sim = cfg.simulation
    generator = torch.Generator().manual_seed(cfg.seed)

    # 1. Random acceleration, smoothed with a moving average so the motion is not jittery.
    fine_dt = sim.fine_dt
    fine_times = torch.arange(0.0, sim.duration + fine_dt / 2, fine_dt)
    N = len(fine_times)
    acceleration = torch.randn(N - 1, 2, generator=generator) * sim.acceleration_std
    kernel = torch.ones(1, 1, 51) / 51
    acceleration = torch.nn.functional.conv1d(acceleration.T[:, None, :], kernel, padding=25)[:, 0, :].T

    # 2. Integrate at the fine time step, adding sudden velocity changes (turns).
    turns = {round(c.time / fine_dt): torch.tensor(c.delta_velocity) for c in sim.velocity_changes}
    positions = torch.zeros(N, 2)
    velocity = torch.tensor(sim.initial_velocity, dtype=torch.float32)
    positions[0] = torch.tensor(sim.initial_position)
    for i in range(1, N):
        velocity = velocity + acceleration[i - 1] * fine_dt
        if i in turns:
            velocity = velocity + turns[i]
        positions[i] = positions[i - 1] + velocity * fine_dt

    # 3. Sample the truth at the (coarser) observation times and add sensor noise.
    observation_times = torch.arange(0.0, sim.duration + sim.observation_dt / 2, sim.observation_dt)
    nearest = torch.argmin(torch.abs(fine_times[:, None] - observation_times[None, :]), dim=0)
    truth = positions[nearest]
    measurements = truth + torch.randn(truth.shape, generator=generator) * sim.observation_noise_std

    # 4. Remove measurements during the gaps, and corrupt a few into outliers.
    observed = torch.ones(len(observation_times), dtype=torch.bool)
    for start, end in sim.missing_intervals:
        observed &= ~((observation_times >= start) & (observation_times <= end))
    if not observed[0]:
        raise ValueError("The first measurement must be observed: it starts the track.")
    is_outlier = torch.zeros_like(observed)
    for t in sim.outlier_indices:
        direction = torch.randn(2, generator=generator)
        measurements[t] += direction / direction.norm() * sim.outlier_scale
        is_outlier[t] = True
    measurements[~observed] = torch.nan

    return TrajectoryData(fine_times, positions, observation_times, truth, measurements, observed, is_outlier)
