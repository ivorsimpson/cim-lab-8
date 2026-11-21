"""Error metrics comparing estimated and true positions."""
import torch


def position_rmse(predicted, target, valid=None):
    """Root mean squared position error (RMSE).

    STUDENT TASK 1a:
    predicted: one row per frame; the first two columns are the x, y position. Any further
               columns (velocity, ...) must be ignored.
    target:    (T, 2) the true x, y position at each frame.
    valid:     optional (T,) True/False per frame; only use rows where it is True.
               If it is None, use every row.

    For each row, find the straight-line distance between the predicted position and
    the true one (hint: torch.linalg.norm(..., dim=1)). Square these distances, take
    their mean, then the square root. Return the result as a tensor.
    If no rows are selected, return torch.tensor(float("nan")).
    """
    raise NotImplementedError("Complete Task 1")


def mean_position_error(predicted, target, valid=None):
    """Mean position error.

    STUDENT TASK 1b: as 1a, but return the plain mean of the straight-line distances
    (no squaring). Compared with RMSE, this is less affected by a few very large errors.
    """
    raise NotImplementedError("Complete Task 1")
