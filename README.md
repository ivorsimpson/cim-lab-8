# CIM Lab 8: Kalman filtering with noisy and missing observations

## Aim

Get a feel for how a Kalman filter tracks a moving object, and for what makes it work well or badly. The filter is written for you. You will write two small pieces of code and one figure, then spend most of the session changing one setting at a time and watching what happens.

## Setup

```bash
uv sync --extra dev
uv run pytest
```

All Python commands must use `uv run`, or activate `.venv/bin/activate` first. Most tests fail until you finish Tasks 1–3; that is expected.

## The idea in one minute

We want to know where an object is, but our sensor is noisy. It reports the position every 0.25 s; we call each report time a **frame**. Each measurement is a little off, a few are badly wrong, and during a **gap** there are no measurements at all. A Kalman filter repeats two steps every frame:

1. **Predict.** "It was here, moving this fast, so now it should be *there*." The filter becomes **less** sure of itself, because the object may not move exactly as expected.
2. **Correct.** A measurement arrives. The filter moves its guess towards it, but only partly, and becomes **more** sure of itself.

How far it moves depends on **who it trusts more**: its own prediction, or the sensor. Two settings control this trust:

| Setting | Plain meaning |
| --- | --- |
| `filter.process_noise_std` | How much the filter expects the object to wander from a straight line. Larger means it trusts its predictions less. |
| `filter.observation_noise_std` | How noisy the filter *believes* the sensor is. Larger means it trusts measurements less. |

The filter also keeps track of its own **uncertainty**: a region where it is confident the object is. In the figures this is drawn in orange. A measurement far outside that region is suspicious, and the **gate** throws such measurements away.

## The code

You only edit the three files at the top of `src/cim_lab_8/`. Everything in `provided/` is supplied, but please read it.

| File | What it does |
| --- | --- |
| `metrics.py` | How far the estimate is from the truth (**Task 1**) |
| `gating.py` | Decides whether to trust each measurement (**Task 2**) |
| `my_plots.py` | Your own figure (**Task 3**) |
| `provided/kalman.py` | The Kalman filter: the model matrices, `predict` and `correct` |
| `provided/tracker.py` | `run_filter`: the predict–correct loop, frame by frame |
| `provided/simulate.py` | Makes the true path and the measurements |
| `provided/plotting.py`, `provided/run.py` | Makes the figures and the summary |
| `provided/learning.py` | Lets PyTorch tune the filter (Experiment 7) |
| `conf/config.yaml` | Every setting, with a comment for each |

## Plan for the session (2 hours)

| Time | Activity |
| --- | --- |
| 0:00–0:15 | Setup; read `provided/kalman.py` and `run_filter` in `provided/tracker.py` |
| 0:15–0:25 | Task 1 |
| 0:25–0:50 | Task 2, then your first run |
| 0:50–1:05 | Task 3 |
| 1:05–1:55 | Experiments 1–6 (about 8 minutes each) |
| 1:55–2:00 | Experiment 7 if you have time |

## Task 1: measuring the error

In `metrics.py`, complete `position_rmse` and `mean_position_error`. Both compare the estimated position with the true position. Instructions are in each function's docstring. Check your work with:

```bash
uv run pytest tests/test_task1_metrics.py
```

## Task 2: the gate

In `gating.py`, complete `gate_and_update`. For each measurement it should:

1. work out how surprising the measurement is, compared with the filter's uncertainty;
2. use it if it is plausible, and ignore it if not;
3. if too many measurements in a row are ignored, assume the filter has lost the object and restart it at the latest measurement.

The docstring gives each step. Check with:

```bash
uv run pytest tests/test_task2_gate.py
```

## Running the filter

Once Tasks 1 and 2 pass, run:

```bash
uv run kf
```

Each run saves its results in its own folder, named after the time and the settings you changed, e.g. `outputs/<date>/14-03-22_default/` or `outputs/<date>/14-05-10_filter.process_noise_std=0.2/`. The last line printed tells you where. Inside:

- **`kalman_trajectory.png`**: the path in 2D:
  - black: the truth, with dots and labels every 2 seconds;
  - thick grey band: the part of the path inside the gap;
  - × marks: measurements, green if the filter used them, red if the gate rejected them;
  - black circles: the deliberately wrong measurements;
  - blue: the filter's estimate;
  - orange ellipses: the filter's uncertainty (it thinks the object is inside 95% of the time);
  - purple stars: track restarts.
- **`uncertainty_and_error.png`**: over time:
  - grey dots: the error of each raw measurement, circled if it was deliberately wrong;
  - blue: the filter's error;
  - orange: the filter's uncertainty, the longest radius of its orange ellipse. Blue should be below orange about 95% of the time;
  - shaded area: the gap;
  - red ticks: measurements the gate rejected. A tick under a circled dot is a correct rejection; one without is a good measurement that was thrown away.
- **`error_histogram.png`**: your figure from Task 3.
- **`noise_sweep.png`**: the filter's error for many combinations of the two trust settings (see Experiment 6).
- **`summary.json`**: the numbers (also printed):

  | Key | Meaning |
  | --- | --- |
  | `sensor_alone.rmse_when_measured` | Typical error of the raw measurements |
  | `filter.rmse_when_measured` | Filter's typical error on the same frames, for a fair comparison |
  | `filter.rmse` | Filter's typical error on every frame, including the gap |
  | `filter.gap_rmse` | Filter's typical error during the gap only |
  | `mean_error` | Like RMSE, but a plain average of the distances |
  | `outliers_used`, `good_measurements_rejected`, `restarts` | Counts of what the gate did |

  RMSE squares the errors before averaging, so a few large errors (like the deliberately wrong measurements) push it up much more than they push up `mean_error`.

To change a setting, add it to the command, e.g. `uv run kf filter.process_noise_std=0.2`. Every setting is listed in `conf/config.yaml`.

**Windows users:** the quotes in some commands below (e.g. `'[30,31,32,33]'`) are for macOS/Linux and PowerShell. In the old `cmd` prompt, use double quotes instead.

## Task 3: your own figure

In `my_plots.py`, complete `plot_error_histogram` with Matplotlib. It shows, frame by frame, how big the errors are for the raw sensor and for the filter, as two overlaid histograms with their means marked. The docstring lists the steps. Until you finish it, `uv run kf` simply skips this figure. Check with:

```bash
uv run pytest tests/test_task3_plot.py
```

Then rerun `uv run kf` and open `error_histogram.png`. Which has the longer tail, and which frames do you think are in it?

Note that the two histograms don't cover exactly the same frames. The filter has an error at every frame, including the gap. The sensor has none in the gap, but its frames include the deliberately wrong measurements. The dashed lines are plain means, so they will look closer together than the RMSE numbers in `summary.json`.

Once it works, try improving it, e.g. a log-scale y-axis (`ax.set_yscale("log")`), or a second panel showing only the frames in the gap.

## Experiments

Change one thing at a time, compare the figures with the default run, and talk it through with your neighbour.

### Experiment 1: watch it work

Look at the figures from the default run (`uv run kf`).

- Is the filter better than the raw sensor? Compare `sensor_alone.rmse_when_measured` with `filter.rmse_when_measured`, and look at your histogram. (`filter.rmse` is higher because it also includes the gap, where the sensor has nothing at all.)
- What happens to the orange uncertainty during the gap (about 5–6 s; the grey band on the path)? What happens when measurements return?
- What does the filter do just after the two sudden turns (around 4.2 s and 8.1 s)?

### Experiment 2: a worse sensor

Make the sensor noisier, and tell the filter the truth about it:

```bash
uv run kf simulation.observation_noise_std=0.8 filter.observation_noise_std=0.8
```

Now make the sensor noisier but **don't** tell the filter:

```bash
uv run kf simulation.observation_noise_std=0.8
```

- Which run is better, and why?
- In the second run, count the red × marks. Why does a filter that trusts the sensor too much end up rejecting *good* measurements?

### Experiment 3: how much should it expect the object to wander?

```bash
uv run kf filter.process_noise_std=0.2
uv run kf filter.process_noise_std=5
```

- With 0.2, what happens at the turns? Do you see purple stars (track restarts)? Where are they, compared with the turns and the gap? (The count of rejections in a row carries on across the gap.) In `uncertainty_and_error.png`, does the blue error stay below the orange uncertainty?
- With 5, how does the blue line look on the straight parts?
- Now switch the restart off and look at the trajectory plot:
  ```bash
  uv run kf filter.process_noise_std=0.2 filter.max_consecutive_rejections=0
  ```
  Why does one rejected measurement lead to all the later ones being rejected?

(`noise_sweep.png` won't change in these runs. It always tries every value from its own list, and only marks which one you chose.)

### Experiment 4: what is the gate worth?

Turn the gate off, then make the wrong measurements much worse:

```bash
uv run kf filter.gate_threshold=inf
uv run kf filter.gate_threshold=inf simulation.outlier_scale=6
uv run kf simulation.outlier_scale=6
```

- What does a single wrong measurement do to the estimate without the gate? How does your histogram's tail change?
- What is the downside of a gate? (Look back at what happened after the turns.)

Now send a burst of four wrong measurements in a row:

```bash
uv run kf simulation.outlier_indices='[30,31,32,33]'
uv run kf simulation.outlier_indices='[30,31,32,33]' filter.max_consecutive_rejections=0
```

- The restart rule helped in Experiment 3. What does it do here?

### Experiment 5: losing the signal

Make the gap longer:

```bash
uv run kf simulation.missing_intervals='[[4.0,7.0]]'
```

- In `uncertainty_and_error.png`, compare how close the blue error gets to the orange uncertainty with the default run. Is the filter still honest about what it does not know?
- Something happens to the object during this gap that the filter never sees. What is it, and why does it matter?

### Experiment 6: tuning

Open `noise_sweep.png` from the default run. Each cell is a complete run of the filter with a different pair of trust settings. The outlined cell is the pair you ran with.

- Find the best (darkest) cell. How does it compare with the outlined one?
- Some cells are worse than the raw sensor (0.72). Remember that these numbers include the gap, where the sensor has no measurement at all.
- Which corner is worst? What does the filter believe in that corner?
- Is the best cell for all frames (left) also best during the gap (right)?
- Try `uv run kf seed=3` (a different random path) and look again. Does the best cell stay put?

Optionally, try a filter that also tracks acceleration:

```bash
uv run kf filter.model=constant_acceleration
```

Does it handle the turns and the gap differently?

### Experiment 7 (if time): let PyTorch tune it

Instead of guessing the two trust settings, PyTorch can adjust them by gradient descent to make the filter as little "surprised" by the measurements as possible:

```bash
uv run kf experiment.fit_noise=true
```

- How close is the learned `observation_noise_std` to the true value (0.30)?
- This learning step has no gate. How might the deliberately wrong measurements affect what it learns?

## Reading

Simon J. D. Prince, *Computer Vision: Models, Learning, and Inference*, chapter 19, covers temporal models and the Kalman filter.

## Scope

This lab does not do image matching or learned tracking. It isolates the state-estimation problem so every assumption and failure is visible.

---

## Appendix: the maths

You do not need this for the lab. It is here if you want to connect the code to the equations.

**State.** `x = [p_x, p_y, v_x, v_y]`: position and velocity. The filter stores a mean `m` (best guess) and covariance `P` (uncertainty). Only position is measured.

**Model.** Over one time step `dt`, position changes by velocity × `dt`:

```
x_t = F x_{t-1} + noise with covariance Q
z_t = H x_t     + noise with covariance R
```

`F`, `H` and `Q` are written out in full in `provided/kalman.py`. `Q` scales with `process_noise_std²`, and `R = observation_noise_std² · I`.

**Predict** (`provided/kalman.py`, `predict`):

```
m⁻ = F m
P⁻ = F P Fᵀ + Q
```

**Innovation** (`innovation`): the surprise and its expected size.

```
r = z − H m⁻
S = H P⁻ Hᵀ + R
```

**Gate** (`gating.py`): the squared Mahalanobis distance

```
d² = rᵀ S⁻¹ r
```

follows a chi-square distribution with 2 degrees of freedom when the model is right (a measure of distance that accounts for the shape of the uncertainty). `d² ≤ 9.21` keeps 99% of genuine measurements.

**Correct** (`correct`):

```
K = P⁻ Hᵀ S⁻¹        Kalman gain
m = m⁻ + K r
P = (I − K H) P⁻
```

If `P⁻` is large compared with `R`, then `K` is close to 1 and the filter jumps to the measurement. If `P⁻` is small compared with `R`, then `K` is close to 0 and the filter mostly ignores it.

**Learning the noise** (`provided/learning.py`): each measurement is Gaussian under the prediction, `z_t ~ N(H m⁻, S)`, so the negative log-likelihood is

```
Σ_t ½ [ rᵀ S⁻¹ r + log det S + 2 log 2π ]
```

This is differentiable in `process_noise_std` and `observation_noise_std`, so PyTorch can minimise it with Adam.

**Constant acceleration model.** The state per axis becomes `[p, v, a]`, so position also changes by acceleration × `dt²/2`. `process_noise_std` then describes random changes in acceleration rather than in velocity.
