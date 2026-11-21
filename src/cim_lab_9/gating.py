"""Deciding whether to trust each measurement (Task 2)."""
import torch
from .provided.kalman import correct, initialise, innovation


def gate_and_update(state, measurement, H, R, rejections, filter_cfg, model):
    """Use a measurement only if it is plausible; restart the track if it seems lost.

    Inputs:  the predicted `state`, the new `measurement`, the model matrices H and R,
             and `rejections`: how many measurements in a row have been rejected so far
             (frames in a gap have no measurement, so they don't change this count).
    Returns: a tuple (state, accepted, rejections, restarted):
             the state to keep, whether this measurement was used (True/False),
             the updated rejection count, and whether the track was restarted (True/False).

    STUDENT TASK 2:
    1. Work out how surprising the measurement is. Start with these two lines:
           r, S = innovation(state, measurement, H, R)
           d2 = r @ torch.linalg.inv(S) @ r
       d2 is the distance from the prediction, measured relative to how far off the filter
       expects measurements to be. About 2 is typical; above 9.21 happens only 1% of the time.
    2. If d2 is at most filter_cfg.gate_threshold, the measurement is plausible: use it, by
       calling `correct`. Return the corrected state, say it was accepted, reset the
       count to 0, no restart.
    3. Otherwise ignore the measurement: the state stays as predicted, and the count goes up by 1.
    4. If the count has reached filter_cfg.max_consecutive_rejections (and that setting is
       above 0; 0 means "never restart"), we have probably lost the object. Restart:
       make a fresh state at this measurement with `initialise(measurement, filter_cfg, model)`,
       but copy the old velocity into it: everything after the first two entries of the
       old state's mean. Return the fresh state, not accepted, count back to 0, restarted.
       (`model` is only needed by `initialise`, to know how big the state is.)

    When you have finished, think: why does the restart keep the old velocity instead of
    starting from zero? And why is a single rejection not enough to restart?
    """
    raise NotImplementedError("Complete Task 2")
