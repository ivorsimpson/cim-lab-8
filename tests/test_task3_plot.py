import matplotlib.axes
import matplotlib.pyplot as plt
import torch
from cim_lab_9.my_plots import plot_error_histogram

SENSOR = torch.tensor([0.3, float("nan"), 0.5, 2.8, float("nan")])
FILTERED = torch.tensor([0.2, 0.4, 0.3, 0.6, 1.0])


def _record_hist_calls(monkeypatch):
    calls = []
    real_hist = matplotlib.axes.Axes.hist

    def recording_hist(self, x, *args, **kwargs):
        calls.append((torch.as_tensor(x), kwargs.get("bins", args[0] if args else None)))
        return real_hist(self, x, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "hist", recording_hist)
    return calls


def test_histogram_is_saved(tmp_path):
    path = tmp_path / "error_histogram.png"
    plot_error_histogram(SENSOR, FILTERED, path)
    assert path.exists() and path.stat().st_size > 1000


def test_two_histograms_without_nan_and_with_shared_bins(tmp_path, monkeypatch):
    calls = _record_hist_calls(monkeypatch)
    plot_error_histogram(SENSOR, FILTERED, tmp_path / "h.png")
    assert len(calls) == 2, "draw one histogram for the sensor and one for the filter"
    for data, bins in calls:
        assert torch.isfinite(data).all(), "remove the NaN entries before plotting"
        assert bins is not None and torch.isfinite(torch.as_tensor(bins)).all(), \
            "bins contain NaN: did you take the max before removing NaN?"
    assert torch.equal(torch.as_tensor(calls[0][1]), torch.as_tensor(calls[1][1])), "use the same bins for both"
    assert sorted(len(data) for data, _ in calls) == [3, 5]


def test_figure_is_closed(tmp_path):
    plt.close("all")
    plot_error_histogram(torch.rand(10), torch.rand(10), tmp_path / "h.png")
    assert plt.get_fignums() == []
