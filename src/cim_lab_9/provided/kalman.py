"""A Kalman filter for tracking a point in 2D, written in plain PyTorch."""
from dataclasses import dataclass
import torch


@dataclass
class KalmanState:
    mean: torch.Tensor        # best guess of the state, e.g. [x, y, vx, vy]
    covariance: torch.Tensor  # how uncertain that guess is


def matrices(model, dt, process_noise_std, observation_noise_std, dtype=torch.float32):
    """Return the four model matrices F, H, Q, R.

    F: how the state moves forward by one time step dt
    H: which parts of the state the sensor measures (only position)
    Q: how much the motion may differ from F (process noise)
    R: how noisy the sensor is (observation noise)
    """
    q = process_noise_std**2
    r = observation_noise_std**2

    if model == "constant_velocity":
        # State: [x, y, vx, vy].  New position = old position + velocity * dt.
        F = [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
        H = [[1, 0, 0, 0],
             [0, 1, 0, 0]]
        # A random acceleration during the step nudges position by dt^2/2 and velocity by dt.
        a, b, c = dt**4 / 4, dt**3 / 2, dt**2
        Q = [[a, 0, b, 0],
             [0, a, 0, b],
             [b, 0, c, 0],
             [0, b, 0, c]]

    elif model == "constant_acceleration":
        # State: [x, y, vx, vy, ax, ay].  Acceleration changes velocity, velocity changes position.
        h = dt**2 / 2
        F = [[1, 0, dt, 0, h, 0],
             [0, 1, 0, dt, 0, h],
             [0, 0, 1, 0, dt, 0],
             [0, 0, 0, 1, 0, dt],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1]]
        H = [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0]]
        # A random jerk (change in acceleration) during the step.
        pp, pv, pa = dt**5 / 20, dt**4 / 8, dt**3 / 6
        vv, va, aa = dt**3 / 3, dt**2 / 2, dt
        Q = [[pp, 0, pv, 0, pa, 0],
             [0, pp, 0, pv, 0, pa],
             [pv, 0, vv, 0, va, 0],
             [0, pv, 0, vv, 0, va],
             [pa, 0, va, 0, aa, 0],
             [0, pa, 0, va, 0, aa]]
    else:
        raise ValueError(f"Unknown model: {model}")

    F = torch.tensor(F, dtype=dtype)
    H = torch.tensor(H, dtype=dtype)
    Q = torch.tensor(Q, dtype=dtype) * q
    R = torch.eye(2, dtype=dtype) * r
    return F, H, Q, R


def initialise(measurement, filter_cfg, model):
    """Start the track at a measurement; velocity (and acceleration) start at zero."""
    size = 4 if model == "constant_velocity" else 6
    mean = torch.zeros(size, dtype=measurement.dtype)
    mean[:2] = measurement
    stds = [filter_cfg.initial_position_std] * 2 + [filter_cfg.initial_velocity_std] * 2
    if size == 6:
        stds += [filter_cfg.initial_acceleration_std] * 2
    covariance = torch.diag(torch.tensor(stds, dtype=measurement.dtype) ** 2)
    return KalmanState(mean, covariance)


def predict(state, F, Q):
    """Prediction: move the estimate forward in time using the motion model.

    Uncertainty grows: the old uncertainty is carried forward (F P F^T) and the
    process noise Q is added because the object may not move exactly as predicted.
    """
    mean = F @ state.mean
    covariance = F @ state.covariance @ F.T + Q
    return KalmanState(mean, covariance)


def innovation(state, measurement, H, R):
    """How far the measurement is from the prediction (r), and how far we expect it to be (S)."""
    r = measurement - H @ state.mean
    S = H @ state.covariance @ H.T + R
    return r, S


def correct(state, measurement, H, R):
    """Correction: move the prediction towards the measurement, by the right amount.

    The Kalman gain K decides how far to move: close to 1 when the prediction is
    uncertain and the sensor is good, close to 0 when the prediction is confident
    and the sensor is noisy.
    """
    r, S = innovation(state, measurement, H, R)
    K = state.covariance @ H.T @ torch.linalg.inv(S)
    mean = state.mean + K @ r
    I = torch.eye(len(state.mean), dtype=state.mean.dtype)
    covariance = (I - K @ H) @ state.covariance   # uncertainty always shrinks
    return KalmanState(mean, covariance)
