import torch
from cim_lab_8.metrics import mean_position_error, position_rmse

# One estimate is at (3, 4), which is 5 away from the truth at (0, 0) in a straight line.
# The other is exactly right. So the distances are 5 and 0.
TARGET = torch.zeros(2, 2)
PREDICTED = torch.tensor([[3.0, 4.0], [0.0, 0.0]])


def _check(value, expected, hint):
    assert isinstance(value, torch.Tensor), f"return a tensor, not {type(value).__name__}"
    assert torch.isclose(value, torch.tensor(expected)), f"got {value.item():.4f}, expected {expected:.4f}. {hint}"


def test_mean_error_uses_straight_line_distance():
    _check(mean_position_error(PREDICTED, TARGET), 2.5,
           "Distances are 5 and 0, so the mean is 2.5. Use the straight-line distance per row, "
           "e.g. torch.linalg.norm(..., dim=1), not the error in x and y separately.")


def test_rmse_uses_straight_line_distance():
    _check(position_rmse(PREDICTED, TARGET), 12.5 ** 0.5,
           "Distances are 5 and 0: squared 25 and 0, mean 12.5, square root 3.536.")


def test_only_position_columns_are_used():
    predicted = torch.tensor([[3.0, 4.0, 100.0, -100.0], [0.0, 0.0, 50.0, 50.0]])
    _check(mean_position_error(predicted, TARGET), 2.5, "Use only the first two columns (x, y), e.g. predicted[:, :2].")
    _check(position_rmse(predicted, TARGET), 12.5 ** 0.5, "Use only the first two columns (x, y), e.g. predicted[:, :2].")


def test_valid_mask_selects_rows():
    target = torch.zeros(3, 2)
    predicted = torch.tensor([[3.0, 4.0], [6.0, 8.0], [0.0, 0.0]])
    valid = torch.tensor([False, True, True])
    hint = "Only rows where valid is True should count (distances 10 and 0 here)."
    _check(mean_position_error(predicted, target, valid), 5.0, hint)
    _check(position_rmse(predicted, target, valid), 50.0 ** 0.5, hint)


def test_empty_mask_gives_nan():
    none = torch.zeros(2, dtype=torch.bool)
    for function in (position_rmse, mean_position_error):
        value = function(torch.ones(2, 2), torch.zeros(2, 2), none)
        assert isinstance(value, torch.Tensor) and torch.isnan(value), \
            "when no rows are selected, return torch.tensor(float('nan'))"
