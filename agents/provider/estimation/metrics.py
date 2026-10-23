"""S2 §4 metrics, in the units the simulator consumes.

Every number here exists to be handed to something else:

* **σ_cost** is the RMSE of ``log(predicted / realized)``. That is precisely
  ``NoisyOracleEstimator.cost_sigma``'s definition, so an S2 measurement drops
  into ``experiments/simulator/estimators.py`` without a unit conversion — which
  is the whole point of deliverable 5.
* **σ_quality** is RMSE in rating points, matching ``quality_sigma``.
* **Spearman ρ** is reported *descriptively, with a CI*, and never as a
  threshold. At n = 10 requirements the 95% interval is roughly ±0.6, so a
  point estimate of 0.4 and a point estimate of 0.0 are the same measurement.
  :func:`spearman_with_ci` therefore always returns the interval alongside the
  estimate, so a caller cannot quote one without the other.
* **Calibration slope** separates "directionally right" from "usable". An
  estimator with ρ = 0.8 and slope 0.1 orders requirements correctly and gets
  every magnitude wrong; ``noisy_oracle`` needs the magnitude.
* **Joint structure** checks the predicted (cost, quality) correlation against
  the realized within-tier +0.82 / +0.56 / +0.75. S1b §4 showed that predicting
  cheap-and-good jobs — a correlation that does not exist in this data — is
  exactly the error that turns cost information into an adverse-selection trap.

Pure stdlib, matching the simulator's own dependency footprint. Nothing here
needs numpy and adding it would make the estimation package harder to run than
the market it feeds.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

#: Bootstrap resamples for the ρ interval. Enough that the reported bounds are
#: stable to ~0.01 across seeds, which is far finer than the interval's own
#: width at n = 10.
BOOTSTRAP_RESAMPLES = 10_000

#: Fixed so a report regenerates identically. The CI is a statement about the
#: data, not about which seed the analyst happened to run.
BOOTSTRAP_SEED = 20260804


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson r, returning 0.0 when either series is constant.

    A constant series has no correlation *defined*, not a correlation of zero;
    0.0 is returned because every caller here treats "no relationship
    measurable" and "no relationship" the same way, and raising would abort a
    whole report over one degenerate tier.
    """
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx, my = statistics.mean(xs), statistics.mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return numerator / denominator if denominator > 0 else 0.0


def _ranks(values: Sequence[float]) -> List[float]:
    """Ranks with ties averaged — required, because predicted multipliers tie."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman ρ as Pearson r on average-tied ranks."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    return pearson(_ranks(xs), _ranks(ys))


@dataclass(frozen=True)
class CorrelationEstimate:
    """ρ with its uncertainty. Both, always — see the module docstring."""

    rho: float
    ci_low: float
    ci_high: float
    n: int
    method: str

    @property
    def spans_zero(self) -> bool:
        return self.ci_low <= 0.0 <= self.ci_high

    def format(self) -> str:
        if self.method != "bootstrap-percentile":
            # A constant predictor (B0) has no rank ordering to correlate. That
            # is not "ρ = 0 with a wide interval", it is "the question does not
            # apply", and printing ±1.00 would invite someone to read it as a
            # measurement.
            return f"n/a ({self.method})"
        return f"{self.rho:+.2f} [{self.ci_low:+.2f}, {self.ci_high:+.2f}]"


def spearman_with_ci(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> CorrelationEstimate:
    """Bootstrap percentile CI for ρ, resampling *pairs*.

    The bootstrap rather than the Fisher-z approximation, because Fisher-z
    assumes bivariate normality and n large enough for the transform to be
    approximately normal; at n = 10 with tied ranks neither holds. The
    percentile bootstrap is not exact either, but it fails in a way that is
    visible (a lumpy, asymmetric interval) rather than one that produces a
    tidy, confidently wrong number.
    """
    n = len(xs)
    if n != len(ys) or n < 3:
        return CorrelationEstimate(0.0, -1.0, 1.0, n, "undefined")

    point = spearman(xs, ys)
    rng = random.Random(seed)
    draws: List[float] = []
    for _ in range(resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_x = [xs[i] for i in indices]
        sample_y = [ys[i] for i in indices]
        # A resample can be entirely one requirement; ρ is then undefined and
        # the draw carries no information, so it is skipped rather than scored
        # as 0.0 (which would bias the interval toward the null).
        if len(set(sample_x)) < 2 or len(set(sample_y)) < 2:
            continue
        draws.append(spearman(sample_x, sample_y))

    if len(draws) < 100:
        return CorrelationEstimate(point, -1.0, 1.0, n, "bootstrap-degenerate")

    draws.sort()
    low = draws[max(0, int(math.floor((alpha / 2.0) * len(draws))))]
    high = draws[min(len(draws) - 1, int(math.ceil((1.0 - alpha / 2.0) * len(draws))))]
    return CorrelationEstimate(point, low, high, n, "bootstrap-percentile")


# ---------------------------------------------------------------------------
# Error in simulator units
# ---------------------------------------------------------------------------

def rmse(values: Sequence[float]) -> float:
    """Root mean square of already-computed residuals.

    Uncentred on purpose. A biased estimator is a worse estimator, and
    centring the residuals first would report the σ it *would* have had if
    someone corrected its bias — a number no arm of this experiment actually
    achieves.
    """
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def log_ratio_residuals(
    predicted: Sequence[float],
    realized: Sequence[float],
) -> List[float]:
    """``log(predicted / realized)`` — ``cost_sigma``'s definition, literally.

    Non-positive values are dropped rather than clamped: a predicted cost of
    zero is a parse failure that should be counted as one upstream, not
    silently turned into a very large but finite error here.
    """
    residuals = []
    for p, r in zip(predicted, realized):
        if p > 0 and r > 0:
            residuals.append(math.log(p / r))
    return residuals


def sigma_cost(predicted: Sequence[float], realized: Sequence[float]) -> float:
    """σ_cost in ``NoisyOracleEstimator.cost_sigma`` units."""
    return rmse(log_ratio_residuals(predicted, realized))


def sigma_quality(predicted: Sequence[float], realized: Sequence[float]) -> float:
    """σ_quality in ``NoisyOracleEstimator.quality_sigma`` units (rating points)."""
    return rmse([p - r for p, r in zip(predicted, realized)])


def mean_bias(predicted: Sequence[float], realized: Sequence[float]) -> float:
    """Mean of ``log(predicted / realized)`` — the part of σ that is offset.

    Reported separately because bias and dispersion have different fixes and
    very different consequences: a systematically-high estimator prices itself
    out of every auction, while a noisy unbiased one merely prices badly.
    """
    residuals = log_ratio_residuals(predicted, realized)
    return statistics.mean(residuals) if residuals else 0.0


def sigma_cost_debiased(
    predicted: Sequence[float], realized: Sequence[float]
) -> float:
    """σ_cost after removing a single constant multiplicative offset.

    **Pre-specified before the full probe was run**, on the strength of a
    two-call smoke test in which both models judged v1's *cheapest* requirement
    to be 35–90% harder than typical. One requirement cannot distinguish a
    level bias from a single miss, so rather than decide later what the number
    meant, both quantities are reported for every arm.

    Why the debiased figure is legitimate and not a way of flattering the LLM:
    correcting a constant offset requires **no task information whatsoever**.
    The agent observes its own realized costs, so an EMA of
    ``predicted / realized`` gives it the constant for free — the same way it
    already learns its tier level. What it cannot learn from history is which
    *particular* job deviates from the mean, and that is exactly what the
    dispersion term measures. The raw figure answers "what happens if you plug
    the multiplier straight in"; the debiased figure answers "what is the
    task-conditional information actually worth". The simulator wants the
    second, deliverable 5 records both, and quoting only one would be
    misleading in whichever direction it was chosen.

    Applied uniformly to every arm, including B0 — whose small positive bias is
    an artefact of using the arithmetic rather than geometric tier mean — so no
    arm gets a correction its competitors do not.
    """
    residuals = log_ratio_residuals(predicted, realized)
    if not residuals:
        return 0.0
    offset = statistics.mean(residuals)
    return rmse([r - offset for r in residuals])


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationFit:
    """Realized regressed on predicted: ``realized = intercept + slope·predicted``.

    Slope 1.0 is perfect magnitude transfer. Slope ≈ 0 with ρ > 0 means the
    estimator ranks requirements correctly but compresses every difference to
    nothing — directionally right, magnitude-useless, and worthless to a
    simulator that needs to know *how much* cheaper the cheap job is.
    """

    slope: float
    intercept: float
    r_squared: float
    n: int


def calibration_fit(
    predicted: Sequence[float],
    realized: Sequence[float],
) -> CalibrationFit:
    """Univariate OLS of realized on predicted."""
    n = len(predicted)
    if n < 2 or n != len(realized):
        return CalibrationFit(0.0, 0.0, 0.0, n)
    mx, my = statistics.mean(predicted), statistics.mean(realized)
    sxx = sum((x - mx) ** 2 for x in predicted)
    if sxx <= 0:
        return CalibrationFit(0.0, my, 0.0, n)
    sxy = sum((x - mx) * (y - my) for x, y in zip(predicted, realized))
    slope = sxy / sxx
    intercept = my - slope * mx
    r = pearson(predicted, realized)
    return CalibrationFit(slope, intercept, r * r, n)


# ---------------------------------------------------------------------------
# Joint structure (S2 §4, last bullet)
# ---------------------------------------------------------------------------

#: Realized within-tier ``corr(cost, quality)`` on dataset v1, from S1b §4.
#: An estimator whose predictions do not reproduce these is predicting a market
#: that does not exist — specifically, cheap-and-good jobs.
REALIZED_WITHIN_TIER_COST_QUALITY = {
    "low": 0.82,
    "medium": 0.56,
    "high": 0.75,
}


@dataclass(frozen=True)
class JointStructure:
    """Predicted vs. realized within-tier cost/quality correlation."""

    tier: str
    predicted_corr: float
    realized_corr: float
    reference_corr: Optional[float]

    @property
    def sign_agrees(self) -> bool:
        return (self.predicted_corr >= 0) == (self.realized_corr >= 0)


def joint_structure(
    tier: str,
    predicted_cost: Sequence[float],
    predicted_quality: Sequence[float],
    realized_cost: Sequence[float],
    realized_quality: Sequence[float],
) -> JointStructure:
    """Compare the predicted cost/quality relationship to the realized one.

    The realized correlation is recomputed from the data in hand rather than
    read from the constant above, so the comparison stays honest if the dataset
    changes; the S1b figure is carried alongside as a cross-check that this
    code and that report are looking at the same thing.
    """
    return JointStructure(
        tier=tier,
        predicted_corr=pearson(predicted_cost, predicted_quality),
        realized_corr=pearson(realized_cost, realized_quality),
        reference_corr=REALIZED_WITHIN_TIER_COST_QUALITY.get(tier),
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

@dataclass
class ArmMetrics:
    """Everything S2 §4 asks for, for one arm (a baseline or one model).

    ``per_tier`` and ``pooled`` are kept separate because they answer different
    questions: pooled σ is what the simulator wants, per-tier σ is what tells
    you *where* an estimator fails — and on v1 the high tier has three times
    the cost spread of the low tier, so a pooled number alone hides most of the
    structure.
    """

    arm: str
    n_cells: int
    sigma_cost_pooled: float
    sigma_cost_debiased_pooled: float
    sigma_quality_pooled: float
    sigma_quality_debiased_pooled: float
    cost_bias_pooled: float
    quality_bias_pooled: float
    sigma_cost_by_tier: Dict[str, float]
    sigma_quality_by_tier: Dict[str, float]
    rho_cost_by_tier: Dict[str, CorrelationEstimate]
    rho_quality_by_tier: Dict[str, CorrelationEstimate]
    calibration_cost_by_tier: Dict[str, CalibrationFit]
    calibration_quality_by_tier: Dict[str, CalibrationFit]
    joint_by_tier: Dict[str, JointStructure]
    parse_failure_rate: float = 0.0

    def summary_row(self) -> Dict[str, object]:
        """Flat dict for the report's σ table."""
        return {
            "arm": self.arm,
            "n_cells": self.n_cells,
            "sigma_cost": round(self.sigma_cost_pooled, 4),
            "sigma_cost_debiased": round(self.sigma_cost_debiased_pooled, 4),
            "sigma_quality": round(self.sigma_quality_pooled, 3),
            "sigma_quality_debiased": round(self.sigma_quality_debiased_pooled, 3),
            "cost_bias": round(self.cost_bias_pooled, 4),
            "quality_bias": round(self.quality_bias_pooled, 3),
            "parse_failure_rate": round(self.parse_failure_rate, 3),
        }


@dataclass(frozen=True)
class Prediction:
    """One arm's prediction for one (requirement, tier) cell.

    Absolute USD and rating points, not multipliers: the arms disagree about
    what a multiplier is relative to (B1 has no notion of one), so they are
    compared after resolution against the tier level. The multiplier is kept
    for the results CSV because it is what the model actually emitted, and the
    protocol freezes the *emitted* format.
    """

    probe_id: str
    tier: str
    predicted_cost_usd: float
    predicted_quality: float
    cost_multiplier: float
    quality_delta: float
    parse_failed: bool = False


def evaluate_arm(
    arm: str,
    predictions: Sequence[Prediction],
    realized_cost: Dict[Tuple[str, str], float],
    realized_quality: Dict[Tuple[str, str], float],
    *,
    tiers: Sequence[str] = ("low", "medium", "high"),
) -> ArmMetrics:
    """Score one arm's predictions against the recorded outcomes.

    Repeats are collapsed to a per-cell mean *before* scoring. Scoring each
    repeat separately and averaging the σs would report the error of a single
    draw, whereas the quantity that matters — and the one a deployed agent
    would use — is the error of the estimate the protocol actually produces,
    which is the mean over its 5 repeats.
    """
    by_cell: Dict[Tuple[str, str], List[Prediction]] = {}
    for prediction in predictions:
        by_cell.setdefault((prediction.probe_id, prediction.tier), []).append(prediction)

    collapsed: Dict[Tuple[str, str], Tuple[float, float]] = {}
    failures = 0
    total = 0
    for key, group in by_cell.items():
        total += len(group)
        failures += sum(1 for p in group if p.parse_failed)
        collapsed[key] = (
            statistics.mean([p.predicted_cost_usd for p in group]),
            statistics.mean([p.predicted_quality for p in group]),
        )

    pooled_pred_cost: List[float] = []
    pooled_real_cost: List[float] = []
    pooled_pred_quality: List[float] = []
    pooled_real_quality: List[float] = []

    sigma_c: Dict[str, float] = {}
    sigma_q: Dict[str, float] = {}
    rho_c: Dict[str, CorrelationEstimate] = {}
    rho_q: Dict[str, CorrelationEstimate] = {}
    cal_c: Dict[str, CalibrationFit] = {}
    cal_q: Dict[str, CalibrationFit] = {}
    joint: Dict[str, JointStructure] = {}

    for tier in tiers:
        keys = sorted(k for k in collapsed if k[1] == tier)
        if not keys:
            continue
        pc = [collapsed[k][0] for k in keys]
        pq = [collapsed[k][1] for k in keys]
        rc = [realized_cost[k] for k in keys]
        rq = [realized_quality[k] for k in keys]

        sigma_c[tier] = sigma_cost(pc, rc)
        sigma_q[tier] = sigma_quality(pq, rq)
        rho_c[tier] = spearman_with_ci(pc, rc)
        rho_q[tier] = spearman_with_ci(pq, rq)
        # Calibration for cost is fit in log space: the estimator's job is to
        # get proportional deviations right, and a level-space slope on v1's
        # 3× tier spread would mostly measure the tier effect.
        cal_c[tier] = calibration_fit(
            [math.log(v) for v in pc], [math.log(v) for v in rc]
        )
        cal_q[tier] = calibration_fit(pq, rq)
        joint[tier] = joint_structure(tier, pc, pq, rc, rq)

        pooled_pred_cost.extend(pc)
        pooled_real_cost.extend(rc)
        pooled_pred_quality.extend(pq)
        pooled_real_quality.extend(rq)

    quality_residuals = [
        p - r for p, r in zip(pooled_pred_quality, pooled_real_quality)
    ]
    quality_offset = statistics.mean(quality_residuals) if quality_residuals else 0.0

    return ArmMetrics(
        arm=arm,
        n_cells=len(collapsed),
        sigma_cost_pooled=sigma_cost(pooled_pred_cost, pooled_real_cost),
        sigma_cost_debiased_pooled=sigma_cost_debiased(pooled_pred_cost, pooled_real_cost),
        sigma_quality_pooled=sigma_quality(pooled_pred_quality, pooled_real_quality),
        sigma_quality_debiased_pooled=rmse(
            [r - quality_offset for r in quality_residuals]
        ),
        cost_bias_pooled=mean_bias(pooled_pred_cost, pooled_real_cost),
        quality_bias_pooled=quality_offset,
        sigma_cost_by_tier=sigma_c,
        sigma_quality_by_tier=sigma_q,
        rho_cost_by_tier=rho_c,
        rho_quality_by_tier=rho_q,
        calibration_cost_by_tier=cal_c,
        calibration_quality_by_tier=cal_q,
        joint_by_tier=joint,
        parse_failure_rate=(failures / total) if total else 0.0,
    )
