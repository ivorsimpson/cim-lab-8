import torch
from omegaconf import OmegaConf
from cim_lab_8.provided.kalman import KalmanState, matrices
from cim_lab_8.gating import gate_and_update
from cim_lab_8.provided.tracker import run_filter
from cim_lab_8.provided.simulate import simulate_trajectory

CFG = OmegaConf.create({"gate_threshold": 9.21, "max_consecutive_rejections": 3, "initial_position_std": 0.6,
                        "initial_velocity_std": 1.0, "initial_acceleration_std": 1.0})


def _setup():
    _, H, _, R = matrices("constant_velocity", 0.25, 1.0, 0.3)
    return KalmanState(torch.tensor([0.0, 0.0, 1.0, 0.5]), torch.eye(4) * 0.1), H, R


def test_plausible_measurement_is_accepted():
    state, H, R = _setup()
    new, accepted, rejections, restarted = gate_and_update(state, torch.tensor([0.1, -0.1]), H, R, 2, CFG,
                                                           "constant_velocity")
    assert accepted, "a measurement close to the prediction should be accepted"
    assert rejections == 0, "after accepting, reset the rejection count to 0"
    assert not restarted
    assert new.mean[0] > 0, "an accepted measurement should be used: call correct(...)"


def test_outlier_is_rejected_and_state_unchanged():
    state, H, R = _setup()
    new, accepted, rejections, restarted = gate_and_update(state, torch.tensor([5.0, 5.0]), H, R, 0, CFG,
                                                           "constant_velocity")
    assert not accepted, "a measurement far from the prediction should be rejected"
    assert rejections == 1, "after a rejection, add 1 to the count"
    assert not restarted, "one rejection (limit is 3) should not restart the track"
    assert torch.equal(new.mean, state.mean) and torch.equal(new.covariance, state.covariance), \
        "a rejected measurement must not change the state"


def test_repeated_rejections_restart_the_track():
    state, H, R = _setup()
    new, accepted, rejections, restarted = gate_and_update(state, torch.tensor([5.0, 5.0]), H, R, 2, CFG,
                                                           "constant_velocity")
    assert restarted, "the 3rd rejection in a row (limit 3) should restart the track"
    assert not accepted and rejections == 0, "a restart is not an acceptance, and resets the count to 0"
    assert torch.allclose(new.mean[:2], torch.tensor([5.0, 5.0])), "restart at the measurement's position"
    assert torch.allclose(new.mean[2:], torch.tensor([1.0, 0.5])), "keep the old velocity when restarting"
    assert torch.allclose(new.covariance, torch.diag(torch.tensor([0.36, 0.36, 1.0, 1.0]))), \
        "use initialise(...) for the restarted state's uncertainty"


def test_restart_can_be_disabled():
    state, H, R = _setup()
    cfg = OmegaConf.merge(CFG, {"max_consecutive_rejections": 0})
    _, _, rejections, restarted = gate_and_update(state, torch.tensor([5.0, 5.0]), H, R, 10, cfg,
                                                  "constant_velocity")
    assert not restarted, "max_consecutive_rejections = 0 means never restart"
    assert rejections == 11


def _rmse(data, result):
    # Computed here rather than with metrics.py, so these tests do not depend on Task 1.
    error = result.means[:, :2] - data.true_observation_positions
    return (error ** 2).sum(dim=1).mean().sqrt().item()


def test_default_run_tracks_and_rejects_outliers(make_cfg):
    cfg = make_cfg()
    data = simulate_trajectory(cfg)
    result = run_filter(data, cfg)
    assert not (result.accepted & data.is_outlier).any(), "the deliberately wrong measurements should be rejected"
    rmse = _rmse(data, result)
    assert rmse < 0.8, f"default run RMSE is {rmse:.2f}; expected about 0.51"


def test_restart_rescues_low_process_noise(make_cfg):
    lost = make_cfg("filter.process_noise_std=0.35", "filter.max_consecutive_rejections=0")
    rescued = make_cfg("filter.process_noise_std=0.35")
    data = simulate_trajectory(lost)
    lost_rmse = _rmse(data, run_filter(data, lost))
    rescued_rmse = _rmse(data, run_filter(data, rescued))
    assert lost_rmse > 3, f"without restarts the track should be lost (RMSE {lost_rmse:.2f}, expected about 4.8)"
    assert rescued_rmse < 1.5, f"restarts should recover the track (RMSE {rescued_rmse:.2f}, expected about 0.9)"
