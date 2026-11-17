"""gta6-twitch-ccu-forecast-01 entry in Python 3.13 (standard library only).

Forecasts Twitch concurrent viewers in the Grand Theft Auto VI category for
hours 2..24 of its release day (UTC), as five quantiles an hour, from the
case's hour-1 print (viewers + channels) and the 38-42 past big-launch curves
in the case's `comparables`.

Method (all fitted at run time from the case itself, nothing hard-coded from
the practice answers):
  1. For every comparable launch, take its release-day 24-hour curve
     (hours 1..24 of its day_start) and form the log-ratio curve
     l_h = ln(viewers(h) / viewers(1)).  Launches whose unlock was one
     single simultaneous global time (no rolling-midnight component) are
     dropped: their hour-1 print is near zero and their ratio curve is a
     different regime from a rolling-midnight console launch.
  2. For each forecast hour h, fit an ordinary least squares regression of
     l_h on [ln(viewers(1)), ln(channels(1))] across the comparables and
     predict the case's l_h from its own hour-1 print.  The regression is
     what carries the launch's scale: a small hour-1 print means most of
     the world has not unlocked yet, so the curve rises much more steeply
     later in the day; a huge hour-1 print means the audience is already
     in and the curve stays flatter.  Early hours (2..6) have near-zero
     slopes -- the pre-US-unlock dip is the same for every launch -- while
     the evening peak hours carry strongly negative slopes.
  3. Blend the regression prediction 75/25 with the comparables' median
     log-ratio at that hour (shrinkage against overfitting), and clamp the
     result to [-1.2, 3.5] in log units.
  4. Spread: the residual standard deviation of the same regression at
     that hour, scaled by 0.7, floored at 0.25 and capped at 1.2 log units.
     Quantiles are q50 * exp(z_tau * s_h) with z the standard normal
     quantiles (-1.645, -0.674, 0, 0.674, 1.645), so the 5%-95% band
     covers about 90% of hours.
  5. Fallback: if the case holds no usable comparable curve at all, fall
     back to persistence (median ratio 0, spread 0.15*sqrt(h-1)), i.e. the
     house baseline's shape, so the entry still returns a valid forecast.

Deterministic: fixed iteration order, no randomness, no wall clock, no
network, no input besides the case file.  Runs in well under a second per
case on one CPU.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only.  Outside the grader's container,
MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Fit hyper-parameters (chosen on the practice cases + leave-one-out over the
# comparables; see dev-log.md in the build directory).
SHRINK = 0.25          # weight of the pool median vs the regression prediction
SPREAD_SCALE = 0.7     # multiplier on the residual standard deviation
SPREAD_FLOOR = 0.25    # minimum log spread per hour
SPREAD_CAP = 1.2       # maximum log spread per hour
LOG_RATIO_MIN = -1.2   # clamp on the predicted median log-ratio
LOG_RATIO_MAX = 3.5
MIN_DAY_HOURS = 20     # a comparable needs at least this many release-day hours


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def median(values: list[float]) -> float:
    """Median of a non-empty list (statistics.median, reimplemented to keep this file self-contained)."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def pstdev(values: list[float]) -> float:
    n = len(values)
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def linfit2(xs1: list[float], xs2: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least squares for y = a + b1*x1 + b2*x2, solved by 3x3 Gaussian elimination.

    A tiny ridge (1e-9 on the diagonal) keeps the solve stable if the two
    predictors are nearly collinear; if the system is still degenerate the
    caller falls back to the pool median (see fit_forecast).
    """
    n = len(ys)
    rows = [[1.0, x1, x2] for x1, x2 in zip(xs1, xs2)]
    ata = [[sum(rows[k][i] * rows[k][j] for k in range(n)) for j in range(3)] for i in range(3)]
    aty = [sum(rows[k][i] * ys[k] for k in range(n)) for i in range(3)]
    for i in range(3):
        ata[i][i] += 1e-9
    for i in range(3):
        piv = max(range(i, 3), key=lambda r: abs(ata[r][i]))
        if abs(ata[piv][i]) < 1e-12:
            raise ValueError("singular regression")
        ata[i], ata[piv] = ata[piv], ata[i]
        aty[i], aty[piv] = aty[piv], aty[i]
        for r in range(i + 1, 3):
            f = ata[r][i] / ata[i][i]
            for c in range(i, 3):
                ata[r][c] -= f * ata[i][c]
            aty[r] -= f * aty[i]
    beta = [0.0, 0.0, 0.0]
    for i in (2, 1, 0):
        beta[i] = (aty[i] - sum(ata[i][j] * beta[j] for j in range(i + 1, 3))) / ata[i][i]
    return beta[0], beta[1], beta[2]


def is_pure_global(comp: dict) -> bool:
    """True when a comparable's unlock was one simultaneous global time.

    A launch with a rolling-midnight component ('rolling' / 'midnight
    local' in its unlock text) is not pure-global even if its PC version
    unlocked globally: its console audience still ramps through the day.
    """
    text = ((comp.get("unlock") or "") + " " + (comp.get("platforms") or "")).lower()
    rolling = ("rolling" in text) or ("midnight local" in text)
    return (not rolling) and (("global" in text) or ("simultaneous" in text))


def day_curve(comp: dict) -> dict[int, tuple[float, float]]:
    """Release-day curve {hour 1..24: (viewers, channels)} from a comparable."""
    day_start = comp.get("day_start") or ""
    prefix = day_start[:10]
    hours: dict[int, tuple[float, float]] = {}
    for row in comp.get("hourly") or []:
        stamp = row.get("from") or ""
        if not stamp.startswith(prefix):
            continue
        hour = int(stamp[11:13]) + 1
        viewers = row.get("viewers")
        if 1 <= hour <= 24 and isinstance(viewers, (int, float)) and viewers > 0:
            channels = row.get("channels")
            if not isinstance(channels, (int, float)) or channels is None:
                channels = 1
            hours[hour] = (float(viewers), float(channels))
    return hours


def pool_curves(case: dict) -> list[dict[int, tuple[float, float]]]:
    """Usable comparable release-day curves, in file order (deterministic)."""
    curves = []
    for comp in case.get("comparables") or []:
        if is_pure_global(comp):
            continue
        hours = day_curve(comp)
        if len(hours) >= MIN_DAY_HOURS and hours.get(1, (0.0, 0.0))[0] > 0:
            curves.append(hours)
    return curves


def fit_forecast(curves: list[dict[int, tuple[float, float]]], p1: float, ch1: float) -> dict[int, tuple[float, float]]:
    """Per hour: (median log-ratio to hour 1, log spread)."""
    log_p1 = math.log(max(1.0, p1))
    log_ch1 = math.log(max(1.0, ch1))
    out: dict[int, tuple[float, float]] = {}
    for h in range(2, 25):
        pts = [
            (math.log(hc[1][0]), math.log(max(1.0, hc[1][1])), math.log(hc[h][0] / hc[1][0]))
            for hc in curves if h in hc
        ]
        if len(pts) < 8:
            # Too few curves carry this hour: fall back to the pool median
            # over the hours that do exist, else persistence.
            ratios = [math.log(hc[h][0] / hc[1][0]) for hc in curves if h in hc]
            med = median(ratios) if ratios else 0.0
            out[h] = (med, SPREAD_FLOOR)
            continue
        xs1 = [p[0] for p in pts]
        xs2 = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        try:
            a, b1, b2 = linfit2(xs1, xs2, ys)
        except ValueError:
            # Degenerate regression (collinear predictors): use the pool median.
            out[h] = (median(ys), SPREAD_FLOOR)
            continue
        pred = a + b1 * log_p1 + b2 * log_ch1
        med = median(ys)
        value = (1.0 - SHRINK) * pred + SHRINK * med
        value = max(LOG_RATIO_MIN, min(LOG_RATIO_MAX, value))
        residuals = [y - (a + b1 * x1 + b2 * x2) for x1, x2, y in pts]
        spread = pstdev(residuals) * SPREAD_SCALE if len(residuals) > 1 else SPREAD_FLOOR
        spread = max(SPREAD_FLOOR, min(SPREAD_CAP, spread))
        out[h] = (value, spread)
    return out


def hour_quantiles(p1: float, med_log_ratio: float, spread: float) -> list[float | int]:
    """The five quantiles for one hour around the median ratio."""
    q50 = p1 * math.exp(med_log_ratio)
    qs: list[float | int] = []
    prev: float | int = 0
    for z in Z:
        q = q50 * math.exp(z * spread)
        q = max(0.01, min(1e9, q))
        q = round2(q)
        # Rounding must not break the non-decreasing order.
        if q < prev:
            q = prev
        qs.append(q)
        prev = q
    return qs


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    ch1 = first.get("channels")
    if not isinstance(ch1, (int, float)) or ch1 is None:
        ch1 = 1
    curves = pool_curves(case)
    if curves:
        fitted = fit_forecast(curves, p1, float(ch1))
    else:
        # No usable comparables: persistence fallback (house baseline shape).
        fitted = {h: (0.0, 0.15 * math.sqrt(h - 1)) for h in range(2, 25)}
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        med_log_ratio, spread = fitted.get(h, (0.0, 0.15 * math.sqrt(max(0, h - 1))))
        hours.append({"hour": h, "q": hour_quantiles(p1, med_log_ratio, spread)})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
