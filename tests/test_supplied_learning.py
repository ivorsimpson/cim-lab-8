import math
import torch
from cim_lab_8.provided.learning import negative_log_likelihood
from cim_lab_8.provided.simulate import simulate_trajectory


def _nll(data, cfg, q, r):
    log_q = torch.tensor(math.log(q), dtype=torch.float64, requires_grad=True)
    log_r = torch.tensor(math.log(r), dtype=torch.float64, requires_grad=True)
    return negative_log_likelihood(log_q, log_r, data, cfg), log_q, log_r


def test_loss_is_a_differentiable_scalar(make_cfg):
    cfg = make_cfg()
    data = simulate_trajectory(cfg)
    loss, log_q, log_r = _nll(data, cfg, 1.0, 0.3)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert log_q.grad != 0 and log_r.grad != 0


def test_two_measurements_match_gaussian_density(make_cfg):
    cfg = make_cfg("simulation.duration=0.25", "simulation.outlier_indices=[]", "simulation.missing_intervals=[]")
    data = simulate_trajectory(cfg)
    q, r, dt = 0.5, 0.3, 0.25
    loss, _, _ = _nll(data, cfg, q, r)
    variance = (cfg.filter.initial_position_std ** 2 + dt ** 2 * cfg.filter.initial_velocity_std ** 2
                + q ** 2 * dt ** 4 / 4 + r ** 2)
    residual = (data.measurements[1] - data.measurements[0]).double()
    expected = 0.5 * (residual @ residual / variance + 2 * math.log(variance) + 2 * math.log(2 * math.pi))
    assert math.isclose(loss.item(), expected.item(), rel_tol=1e-6)


def test_prefers_sensible_noise(make_cfg):
    cfg = make_cfg()
    data = simulate_trajectory(cfg)
    assert _nll(data, cfg, 1.0, 0.3)[0] < _nll(data, cfg, 1.0, 5.0)[0]
