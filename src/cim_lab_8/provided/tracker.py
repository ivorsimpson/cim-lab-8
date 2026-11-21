"""Run the Kalman filter over a sequence of (possibly missing or wrong) measurements."""
from dataclasses import dataclass
import torch
from ..gating import gate_and_update
from .kalman import initialise, matrices, predict


@dataclass
class TrackResult:
    means: torch.Tensor          # (T, D) estimated state at each frame
    covariances: torch.Tensor    # (T, D, D) uncertainty of each estimate
    accepted: torch.Tensor       # (T,) True where the measurement was used
    restarted: torch.Tensor      # (T,) True where the track was restarted


def run_filter(data, cfg, model=None, process_noise_std=None, observation_noise_std=None):
    """Filter every frame in `data`. The optional arguments override the config (used by the sweep)."""
    model = model or cfg.filter.model
    q = cfg.filter.process_noise_std if process_noise_std is None else process_noise_std
    r = cfg.filter.observation_noise_std if observation_noise_std is None else observation_noise_std
    dt = float(data.observation_times[1] - data.observation_times[0])
    F, H, Q, R = matrices(model, dt, q, r)

    T = len(data.observation_times)
    state = initialise(data.measurements[0], cfg.filter, model)  # start at the first measurement
    means = torch.zeros(T, len(state.mean))
    covariances = torch.zeros(T, len(state.mean), len(state.mean))
    accepted = torch.zeros(T, dtype=torch.bool)
    restarted = torch.zeros(T, dtype=torch.bool)
    accepted[0] = True
    rejections = 0

    for t in range(T):
        if t > 0:
            state = predict(state, F, Q)                     # 1. predict
            if data.observed[t]:                             # 2. correct, if we have a measurement
                state, accepted[t], rejections, restarted[t] = gate_and_update(
                    state, data.measurements[t], H, R, rejections, cfg.filter, model)
        means[t] = state.mean
        covariances[t] = state.covariance
    return TrackResult(means, covariances, accepted, restarted)
