import torch
from cim_lab_9.provided.kalman import KalmanState, correct, matrices, predict


def test_supplied_prediction_moves_with_velocity():
    F, _, Q, _ = matrices("constant_velocity", 0.5, 0.0, 1.0)
    state = KalmanState(torch.tensor([1.0, 2.0, 4.0, -2.0]), torch.eye(4))
    assert torch.allclose(predict(state, F, Q).mean, torch.tensor([3.0, 1.0, 4.0, -2.0]))


def test_supplied_matrices_have_expected_shapes():
    F, H, Q, R = matrices("constant_acceleration", 0.25, 0.3, 0.4)
    assert F.shape == (6, 6) and H.shape == (2, 6) and Q.shape == (6, 6) and R.shape == (2, 2)


def test_correction_matches_scalar_formula():
    # Each axis: prior variance 4, measurement variance 1, so the gain is 4 / (4 + 1) = 0.8.
    _, H, _, R = matrices("constant_velocity", 1.0, 0.1, 1.0)
    state = KalmanState(torch.zeros(4), torch.eye(4) * 4)
    new = correct(state, torch.tensor([5.0, -10.0]), H, R)
    assert torch.allclose(new.mean, torch.tensor([4.0, -8.0, 0.0, 0.0]))
    assert torch.allclose(new.covariance.diagonal(), torch.tensor([0.8, 0.8, 4.0, 4.0]))


def test_correction_matches_textbook_with_correlations():
    torch.manual_seed(0)
    _, H, _, R = matrices("constant_acceleration", 0.25, 0.3, 0.4, dtype=torch.float64)
    A = torch.randn(6, 6, dtype=torch.float64)
    P = A @ A.T + torch.eye(6, dtype=torch.float64)
    m, z = torch.randn(6, dtype=torch.float64), torch.randn(2, dtype=torch.float64)
    K = P @ H.T @ torch.linalg.inv(H @ P @ H.T + R)
    new = correct(KalmanState(m, P), z, H, R)
    assert torch.allclose(new.mean, m + K @ (z - H @ m))
    assert torch.allclose(new.covariance, (torch.eye(6, dtype=torch.float64) - K @ H) @ P)
