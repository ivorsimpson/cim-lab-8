"""Your own figure (Task 3)."""
import matplotlib.pyplot as plt
import torch


def plot_error_histogram(sensor_errors, filter_errors, path):
    """Compare how big the errors are, frame by frame, for the raw sensor and for the filter.

    sensor_errors: (T,) distance from each measurement to the truth; NaN in the gap (no measurement)
    filter_errors: (T,) distance from the filter's estimate to the truth at every frame
    path:          where to save the figure (a .png file)

    STUDENT TASK 3: draw both distributions on one set of axes and save the figure.
    1. Remove the NaN entries from sensor_errors (hint: torch.isnan). Don't skip this: the max
       of a tensor containing NaN is NaN, and Matplotlib then silently draws an empty plot.
    2. Make one set of bin edges shared by both histograms, e.g.
       torch.linspace(0, largest_error, 30), so the bars line up.
    3. Draw both with ax.hist(..., bins=..., alpha=0.5, label=...). Matplotlib accepts tensors.
    4. Mark each mean with a vertical dashed line (ax.axvline) in the same colour.
    5. Label the axes, add a title and a legend, then fig.savefig(path) and plt.close(fig).
    """
    raise NotImplementedError("Complete Task 3")
