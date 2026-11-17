"""gta6-twitch-ccu-forecast-01 — entry v2 (Python 3.13, standard library only).

Forecasts Twitch concurrent viewers for Grand Theft Auto VI on its release
day (UTC), hours 2..24, as five quantiles per hour, from the case's hour-1
print (viewers + channels) and the release-day curves of past big launches
in the case's `comparables`.

Why v2 differs from a plain per-hour regression: the hour-1 print alone
cannot tell a rolling-midnight launch from a simultaneous-global one, and
the two regimes have very different days. A rolling launch unlocks region
by region, so at 00:00Z only a few timezones are live and the curve jumps
when the US East coast unlocks at hour 5, then keeps climbing into the
US evening; a simultaneous launch is live everywhere at once and stays
flat. Pooling both regimes without telling them apart drags every
rolling forecast down — on the practice cases the plain model underpredicts
hours 5..7 by about half a log unit. So v2 fits, per forecast hour, a
weighted regression of each comparable's log-ratio ln(v_h / v_1) on
[ln(v_1), ln(ch_1), is_rolling], with rolling launches weighted double,
predicts at the case's own hour-1 print and its own unlock class (read
from the case notes, the same text the comparables carry), shrinks 35%
toward the rolling launches' median log-ratio at that hour, and spreads
the quantiles by the weighted residual dispersion. Pure simultaneous-global
launches are still dropped from the pool entirely: their hour-1 print is
near zero and their ratio curve belongs to another regime.

Quantiles are q50 * exp(z_tau * s_h) with z the standard normal quantiles,
clamped to (0, 1e9] and rounded half-up to 2 decimals like the house
baseline. Fallbacks keep the output valid: too few curves at an hour fall
back to the rolling-pool median, a degenerate fit falls back to the median,
and a case with no usable comparable at all falls back to persistence.
Deterministic: fixed iteration order, no randomness, no wall clock, no
network. Runs in well under a second per case on one CPU.

Grader: `python3 -P -s /bot/main.py` in python:3.13-slim. Outside the
grader, MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in for /input and
/output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Fit hyper-parameters (practice + leave-one-out over the comparables;
# see dev-log-v2.md in the build directory).
SHRINK = 0.35            # weight of the rolling-pool median vs the regression
SPREAD_SCALE = 0.55      # multiplier on the weighted residual SD
SPREAD_FLOOR = 0.30      # minimum log spread per hour
SPREAD_CAP = 1.2         # maximum log spread per hour
ROLLING_WEIGHT = 2.0     # weight of a rolling-midnight curve in the fit
LOG_RATIO_MIN = -1.2     # clamp on the predicted median log-ratio
LOG_RATIO_MAX = 3.5
MIN_DAY_HOURS = 20       # a comparable needs this many release-day hours
MIN_FIT_POINTS = 8       # else fall back to the rolling-pool median


def round2(x):
    """Round half-up to 2 decimals; whole numbers come back as ints."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value.is_integer() else value


def median_of(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def unlock_kind(text):
    """'rolling' | 'global' | 'other' from an unlock description.

    A launch counts as rolling when its text mentions a rolling or
    midnight-local unlock, even if its PC version unlocked globally at
    the same time: the console audience still ramps through the day.
    """
    low = (text or "").lower()
    if "rolling" in low or "midnight local" in low or "12 am local" in low:
        return "rolling"
    if "global" in low or "simultaneous" in low:
        return "global"
    return "other"


def release_day_curve(comp):
    """{hour 1..24: (viewers, channels)} for one comparable's release day."""
    day = (comp.get("day_start") or "")[:10]
    curve = {}
    for row in comp.get("hourly") or []:
        stamp = row.get("from") or ""
        if not stamp.startswith(day):
            continue
        hour = int(stamp[11:13]) + 1
        viewers = row.get("viewers")
        if 1 <= hour <= 24 and isinstance(viewers, (int, float)) and viewers > 0:
            channels = row.get("channels")
            if not isinstance(channels, (int, float)) or channels is None:
                channels = 1
            curve[hour] = (float(viewers), float(channels))
    return curve


def usable_pool(case):
    """(curves, kinds) for comparables in file order, pure-global dropped."""
    curves, kinds = [], []
    for comp in case.get("comparables") or []:
        kind = unlock_kind((comp.get("unlock") or "") + " " + (comp.get("platforms") or ""))
        if kind == "global":
            continue
        curve = release_day_curve(comp)
        if len(curve) >= MIN_DAY_HOURS and curve.get(1, (0.0, 0.0))[0] > 0:
            curves.append(curve)
            kinds.append(kind)
    return curves, kinds


def solve_linear(matrix, rhs):
    """Gaussian elimination with partial pivoting; raises on singularity."""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            raise ValueError("singular system")
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(col + 1, n):
            factor = aug[r][col] / aug[col][col]
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        out[i] = (aug[i][n] - sum(aug[i][j] * out[j] for j in range(i + 1, n))) / aug[i][i]
    return out


def weighted_fit(rows, targets, weights):
    """Weighted least squares for y ~ rows (each row already includes the 1s)."""
    k = len(rows[0])
    ata = [[sum(weights[t] * rows[t][i] * rows[t][j] for t in range(len(rows)))
            for j in range(k)] for i in range(k)]
    aty = [sum(weights[t] * rows[t][i] * targets[t] for t in range(len(rows)))
           for i in range(k)]
    for i in range(k):
        ata[i][i] += 1e-9
    return solve_linear(ata, aty)


def fit_hour(curves, kinds, hour, log_p1, log_ch1, case_rolling):
    """(median log-ratio, log spread) for one forecast hour."""
    rows, targets, weights, ratios = [], [], [], []
    rolling_ratios = []
    for curve, kind in zip(curves, kinds):
        if hour not in curve:
            continue
        log_ratio = math.log(curve[hour][0] / curve[1][0])
        ratios.append(log_ratio)
        if kind == "rolling":
            rolling_ratios.append(log_ratio)
        rows.append([1.0, math.log(curve[1][0]), math.log(max(1.0, curve[1][1])),
                     1.0 if kind == "rolling" else 0.0])
        targets.append(log_ratio)
        weights.append(ROLLING_WEIGHT if kind == "rolling" else 1.0)
    shrink_target = (median_of(rolling_ratios) if len(rolling_ratios) >= 3
                     else median_of(ratios))
    if len(rows) < MIN_FIT_POINTS:
        return shrink_target, SPREAD_FLOOR
    try:
        beta = weighted_fit(rows, targets, weights)
    except ValueError:
        return shrink_target, SPREAD_FLOOR
    case_row = [1.0, log_p1, log_ch1, 1.0 if case_rolling else 0.0]
    prediction = sum(b * x for b, x in zip(beta, case_row))
    value = (1.0 - SHRINK) * prediction + SHRINK * shrink_target
    value = max(LOG_RATIO_MIN, min(LOG_RATIO_MAX, value))
    residuals = [y - sum(b * x for b, x in zip(beta, row)) for row, y in zip(rows, targets)]
    total_w = sum(weights)
    mean_res = sum(w * r for w, r in zip(weights, residuals)) / total_w
    variance = sum(w * (r - mean_res) ** 2 for w, r in zip(weights, residuals)) / total_w
    spread = math.sqrt(variance) * SPREAD_SCALE
    spread = max(SPREAD_FLOOR, min(SPREAD_CAP, spread))
    return value, spread


def hour_row(p1, med_log_ratio, spread):
    """The five quantiles for one hour, non-decreasing and in (0, 1e9]."""
    centre = p1 * math.exp(med_log_ratio)
    quantiles = []
    previous = 0
    for z in Z:
        q = centre * math.exp(z * spread)
        q = max(0.01, min(1e9, q))
        q = round2(q)
        if q < previous:
            q = previous
        quantiles.append(q)
        previous = q
    return quantiles


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    channels = first.get("channels")
    ch1 = float(channels) if isinstance(channels, (int, float)) and channels else 1.0
    case_rolling = unlock_kind(case.get("notes") or "") == "rolling"
    curves, kinds = usable_pool(case)
    log_p1, log_ch1 = math.log(p1), math.log(max(1.0, ch1))
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        if curves:
            med_log_ratio, spread = fit_hour(curves, kinds, h, log_p1, log_ch1, case_rolling)
        else:
            med_log_ratio, spread = 0.0, 0.15 * math.sqrt(h - 1)
        hours.append({"hour": h, "q": hour_row(p1, med_log_ratio, spread)})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
