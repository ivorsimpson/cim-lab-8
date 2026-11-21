"""Let PyTorch choose the noise levels: gradient descent on how surprised the filter is."""
import math
import torch
from .kalman import correct, initialise, innovation, matrices, predict


def negative_log_likelihood(log_q, log_r, data, cfg):
    """Total 'surprise' of the filter at the measurements, for noise levels exp(log_q), exp(log_r).

    A good filter predicts each measurement well AND is honest about how uncertain it is.
    Lower is better. Everything here is differentiable, so PyTorch can compute gradients.
    There is no gate, so outliers count too.
    """
    dt = float(data.observation_times[1] - data.observation_times[0])
    F, H, Q1, R1 = matrices(cfg.filter.model, dt, 1.0, 1.0, dtype=torch.float64)
    Q = torch.exp(2 * log_q) * Q1
    R = torch.exp(2 * log_r) * R1
    z = data.measurements.double()
    state = initialise(z[0], cfg.filter, cfg.filter.model)
    loss = torch.zeros((), dtype=torch.float64)
    for t in range(1, len(z)):
        state = predict(state, F, Q)
        if data.observed[t]:
            r, S = innovation(state, z[t], H, R)
            # Negative log of the Gaussian density of the measurement under the prediction.
            loss = loss + 0.5 * (r @ torch.linalg.inv(S) @ r + torch.logdet(S) + 2 * math.log(2 * math.pi))
            state = correct(state, z[t], H, R)
    return loss


def fit_noise(data, cfg, steps=150, learning_rate=0.05):
    """Fit both noise levels with Adam, starting from the configured values."""
    log_q = torch.tensor(math.log(cfg.filter.process_noise_std), dtype=torch.float64, requires_grad=True)
    log_r = torch.tensor(math.log(cfg.filter.observation_noise_std), dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.Adam([log_q, log_r], lr=learning_rate)
    for step in range(steps):
        optimiser.zero_grad()
        loss = negative_log_likelihood(log_q, log_r, data, cfg)
        loss.backward()
        optimiser.step()
        if step % 25 == 0 or step == steps - 1:
            print(f"step {step:3d}  surprise {loss.item():8.2f}  "
                  f"process noise {log_q.exp().item():.3f}  observation noise {log_r.exp().item():.3f}")
    return log_q.exp().item(), log_r.exp().item()
