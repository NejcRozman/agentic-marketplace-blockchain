"""The controls the LLM has to beat (S2 §3).

| arm | what it knows | what it decides |
|---|---|---|
| **B0** | the tier mean | whether the LLM has *any* task signal |
| **B1** | prompt chars, PDF count, prompt count | whether "semantic inference" is `len()` |
| **B1b** | B1 + mechanically counted paper mentions | whether it is `len()` plus a regex |

B1b is the one that matters. Cross-document breadth is the strongest single
predictor on v1 (`corr` ≈ +0.34 against log medium-tier cost) and it is
countable without understanding a single sentence. An LLM that beats B1 but not
B1b has demonstrated pattern-matching over identifiers, which is not the
paper's claim.

**Everything reported is leave-one-requirement-out.** Fitting three coefficients
on ten points and reporting the in-sample fit would produce an R² that looks
like a result and generalises to nothing; on v1 the in-sample/LOO gap is the
single largest number in the whole baseline analysis. :func:`fit_report` returns
both so the gap is visible rather than assumed away.

LOO is by *requirement*, not by cell. The three tiers of one requirement share
its text, so holding out a cell while training on its two siblings would leak
the held-out requirement's difficulty through the back door — the exact failure
mode S5's train/test split exists to prevent, reproduced in miniature.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from agents.provider.estimation import dataset as ds
from agents.provider.estimation.metrics import Prediction

#: Ridge penalty applied to the normal equations. Not a tuning knob — it is
#: numerical insurance so a fold whose design matrix is near-singular returns a
#: shrunken coefficient instead of an exception. At 1e-8 it is far too small to
#: change any fit that was well posed to begin with.
RIDGE_EPSILON = 1e-8


# ---------------------------------------------------------------------------
# Minimal OLS
# ---------------------------------------------------------------------------

def _solve(matrix: List[List[float]], rhs: List[float]) -> List[float]:
    """Gaussian elimination with partial pivoting.

    Hand-rolled to keep the estimation package on the stdlib, like the
    simulator. The systems here are at most 6×6.
    """
    n = len(rhs)
    augmented = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(augmented[r][col]))
        if abs(augmented[pivot][col]) < 1e-14:
            # Structurally singular even after ridging: the column carries no
            # information in this fold. Zeroing its coefficient is the honest
            # answer — the feature cannot be used, so it is not used.
            continue
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        pivot_value = augmented[col][col]
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col] / pivot_value
            if factor == 0.0:
                continue
            for k in range(col, n + 1):
                augmented[row][k] -= factor * augmented[col][k]
    solution = []
    for i in range(n):
        denominator = augmented[i][i]
        solution.append(augmented[i][n] / denominator if abs(denominator) > 1e-14 else 0.0)
    return solution


@dataclass(frozen=True)
class LinearModel:
    """An intercept plus coefficients on the features that survived screening."""

    intercept: float
    coefficients: Dict[str, float]
    dropped: Tuple[str, ...]

    def predict(self, features: Dict[str, float]) -> float:
        return self.intercept + sum(
            weight * features[name] for name, weight in self.coefficients.items()
        )


def fit_linear(
    rows: Sequence[Dict[str, float]],
    targets: Sequence[float],
    feature_names: Sequence[str],
    *,
    ridge_lambda: float = 0.0,
) -> LinearModel:
    """OLS with zero-variance features screened out first.

    ``ridge_lambda`` adds a **statistical** shrinkage penalty, distinct from
    :data:`RIDGE_EPSILON`, which only exists to keep the Gram matrix invertible.
    It is applied to **standardised** columns and the coefficients are unscaled
    afterwards, because the raw columns are on wildly different scales
    (``total_prompt_chars`` is in the thousands, ``prompt_count`` is single
    digits) and an unstandardised penalty would shrink the small-scale
    coefficients almost exclusively. Zero — the default — leaves every existing
    caller's arithmetic byte-for-byte unchanged (S7 §B.2).

    The screen is not cosmetic. On dataset v1 ``source_pdf_count`` is 5 and
    ``prompt_count`` is 10 for every requirement, so two of B1's three columns
    are constant. Left in, they are collinear with the intercept and the fitted
    coefficients are arbitrary — a reader would see a "coefficient on PDF count"
    that means nothing. Screened out and reported in ``dropped``, the situation
    is legible: B1 on v1 *is* a univariate regression on prompt length.
    """
    usable: List[str] = []
    dropped: List[str] = []
    for name in feature_names:
        column = [row[name] for row in rows]
        if len(column) > 1 and statistics.pstdev(column) > 0.0:
            usable.append(name)
        else:
            dropped.append(name)

    if not rows or not usable:
        mean = statistics.mean(targets) if targets else 0.0
        return LinearModel(mean, {}, tuple(dropped))

    # Centre the columns so the intercept is the target mean and the ridge
    # penalty never touches it.
    means = {name: statistics.mean([row[name] for row in rows]) for name in usable}
    design = [[row[name] - means[name] for name in usable] for row in rows]
    target_mean = statistics.mean(targets)
    centred_targets = [t - target_mean for t in targets]

    k = len(usable)

    # Standardise only when a statistical penalty is being applied: with
    # ridge_lambda = 0 the scales below are all 1.0 and the arithmetic is
    # exactly what it was before.
    if ridge_lambda > 0.0:
        scales = [
            statistics.pstdev([row[name] for row in rows]) or 1.0 for name in usable
        ]
        design = [[value / scales[i] for i, value in enumerate(row)] for row in design]
    else:
        scales = [1.0] * k

    gram = [
        [
            sum(design[r][i] * design[r][j] for r in range(len(design)))
            + ((RIDGE_EPSILON + ridge_lambda) if i == j else 0.0)
            for j in range(k)
        ]
        for i in range(k)
    ]
    moment = [
        sum(design[r][i] * centred_targets[r] for r in range(len(design)))
        for i in range(k)
    ]
    weights = _solve(gram, moment)

    # Back to the original units, so the caller never sees the standardisation.
    weights = [w / scales[i] for i, w in enumerate(weights)]

    coefficients = dict(zip(usable, weights))
    intercept = target_mean - sum(
        coefficients[name] * means[name] for name in usable
    )
    return LinearModel(intercept, coefficients, tuple(dropped))


# ---------------------------------------------------------------------------
# Baseline arms
# ---------------------------------------------------------------------------

@dataclass
class TierMeanBaseline:
    """B0 — task-blind. Predicts the tier mean for every requirement.

    This is not a strawman: it is exactly what the live agent's EMA converges
    to, so B0 is the incumbent estimator, and "the LLM beats B0" is the minimum
    claim that would justify making an API call before every bid.
    """

    name: str = "B0_tier_mean"

    def predict(self, data: ds.ProbeDataset) -> List[Prediction]:
        predictions: List[Prediction] = []
        for requirement, tier in data.cells():
            predictions.append(
                Prediction(
                    probe_id=requirement.probe_id,
                    tier=tier,
                    predicted_cost_usd=data.tier_cost_level(tier),
                    predicted_quality=data.tier_quality_level(tier),
                    cost_multiplier=1.0,
                    quality_delta=0.0,
                )
            )
        return predictions


@dataclass
class RegressionBaseline:
    """B1 / B1b — a regression on mechanically countable features.

    Targets are expressed as deviations from the tier level (log-ratio for
    cost, points for quality), pooled across tiers with the level removed. That
    is one coefficient set for all three tiers rather than three fits of ten
    points each — a deliberate trade: the pooled fit assumes a requirement that
    is 10% expensive at the low tier is 10% expensive at the high tier, which
    is an assumption, but three separate 10-point fits would have no power to
    check it anyway.
    """

    feature_names: Tuple[str, ...]
    name: str
    leave_one_out: bool = True

    def predict(
        self,
        data: ds.ProbeDataset,
        *,
        leave_one_out: Optional[bool] = None,
    ) -> List[Prediction]:
        held_out_mode = self.leave_one_out if leave_one_out is None else leave_one_out
        features = {
            r.probe_id: ds.mechanical_features(r.visible) for r in data.requirements
        }
        cost_targets, quality_targets = self._targets(data)

        predictions: List[Prediction] = []
        for requirement, tier in data.cells():
            held_out = requirement.probe_id if held_out_mode else None
            cost_model = self._fit(features, cost_targets, held_out)
            quality_model = self._fit(features, quality_targets, held_out)

            row = features[requirement.probe_id]
            multiplier = math.exp(cost_model.predict(row))
            delta = quality_model.predict(row)
            predictions.append(
                Prediction(
                    probe_id=requirement.probe_id,
                    tier=tier,
                    predicted_cost_usd=data.tier_cost_level(tier) * multiplier,
                    predicted_quality=data.tier_quality_level(tier) + delta,
                    cost_multiplier=multiplier,
                    quality_delta=delta,
                )
            )
        return predictions

    # -- internals ---------------------------------------------------------

    def _targets(
        self, data: ds.ProbeDataset
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """Per-requirement lists of tier-relative deviations, pooled over tiers."""
        cost: Dict[str, List[float]] = {}
        quality: Dict[str, List[float]] = {}
        for requirement, tier in data.cells():
            cost.setdefault(requirement.probe_id, []).append(
                math.log(requirement.realized_cost(tier) / data.tier_cost_level(tier))
            )
            quality.setdefault(requirement.probe_id, []).append(
                requirement.realized_quality(tier) - data.tier_quality_level(tier)
            )
        return cost, quality

    def _fit(
        self,
        features: Dict[str, Dict[str, float]],
        targets: Dict[str, List[float]],
        held_out: str | None,
    ) -> LinearModel:
        rows: List[Dict[str, float]] = []
        values: List[float] = []
        for probe_id, deviations in targets.items():
            if probe_id == held_out:
                continue
            for deviation in deviations:
                rows.append(features[probe_id])
                values.append(deviation)
        return fit_linear(rows, values, self.feature_names)


# ---------------------------------------------------------------------------
# S5 §2 — fitted on dev, evaluated on held-out
# ---------------------------------------------------------------------------

@dataclass
class SplitRegressionBaseline:
    """B1 / B1b as S5 runs them: coefficients from dev, predictions on held-out.

    The LOO machinery above answers "what would this regression deliver on a
    requirement it has not seen", approximated by resampling ten points. v2 does
    not need the approximation — it has a real 20/10 split, so the baseline is
    fitted once on dev and applied to held-out unchanged. That is both the
    honest comparison and the stronger control: B1b gets all 20 dev
    requirements to fit five coefficients, where LOO on v1 gave it nine.

    ``per_tier`` implements S3's carry-forward. :class:`RegressionBaseline`
    pools the three tiers into one coefficient set, which assumes a requirement
    that is 10% expensive at the low tier is 10% expensive at the high tier. On
    v1 that was untestable; on v2 the S3 report measured pooling as costing
    0.0210 σ, so the per-tier fit is available and is what the report leads
    with. Both are run — if the pooled fit wins, that is evidence the
    interaction is weak, and it should be visible rather than assumed away.
    """

    feature_names: Tuple[str, ...]
    name: str
    per_tier: bool = False

    def fit(self, train: ds.ProbeDataset) -> Dict[str, Tuple[LinearModel, LinearModel]]:
        """One (cost, quality) model per tier, or one shared pair under pooling.

        Keyed by tier either way, so :meth:`predict` has a single lookup path
        and the pooled variant is simply the same model stored three times.
        """
        features = {
            r.probe_id: ds.mechanical_features(r.visible) for r in train.requirements
        }
        models: Dict[str, Tuple[LinearModel, LinearModel]] = {}

        if not self.per_tier:
            rows, cost_values, quality_values = self._rows(train, features, tier=None)
            shared = (
                fit_linear(rows, cost_values, self.feature_names),
                fit_linear(rows, quality_values, self.feature_names),
            )
            return {tier: shared for tier in train.tiers}

        for tier in train.tiers:
            rows, cost_values, quality_values = self._rows(train, features, tier=tier)
            models[tier] = (
                fit_linear(rows, cost_values, self.feature_names),
                fit_linear(rows, quality_values, self.feature_names),
            )
        return models

    def predict(
        self,
        train: ds.ProbeDataset,
        evaluate: ds.ProbeDataset,
    ) -> List[Prediction]:
        """Fit on ``train``, predict every cell of ``evaluate``.

        ``evaluate`` supplies the tier levels the multiplier resolves against.
        When it is a held-out slice built with ``levels_from=dev`` those levels
        are dev's, so nothing about the held-out outcomes enters the prediction
        — not the coefficients, and not the level either.
        """
        models = self.fit(train)
        predictions: List[Prediction] = []
        for requirement, tier in evaluate.cells():
            if tier not in models:
                continue
            cost_model, quality_model = models[tier]
            row = ds.mechanical_features(requirement.visible)
            multiplier = math.exp(cost_model.predict(row))
            delta = quality_model.predict(row)
            predictions.append(
                Prediction(
                    probe_id=requirement.probe_id,
                    tier=tier,
                    predicted_cost_usd=evaluate.tier_cost_level(tier) * multiplier,
                    predicted_quality=evaluate.tier_quality_level(tier) + delta,
                    cost_multiplier=multiplier,
                    quality_delta=delta,
                )
            )
        return predictions

    # -- internals ---------------------------------------------------------

    def _rows(
        self,
        train: ds.ProbeDataset,
        features: Dict[str, Dict[str, float]],
        *,
        tier: Optional[str],
    ) -> Tuple[List[Dict[str, float]], List[float], List[float]]:
        """Design matrix and both targets, optionally restricted to one tier."""
        rows: List[Dict[str, float]] = []
        cost: List[float] = []
        quality: List[float] = []
        for requirement, cell_tier in train.cells():
            if tier is not None and cell_tier != tier:
                continue
            rows.append(features[requirement.probe_id])
            cost.append(
                math.log(
                    requirement.realized_cost(cell_tier)
                    / train.tier_cost_level(cell_tier)
                )
            )
            quality.append(
                requirement.realized_quality(cell_tier)
                - train.tier_quality_level(cell_tier)
            )
        return rows, cost, quality


def split_baselines() -> List[SplitRegressionBaseline]:
    """The S5 mechanical controls, in report order.

    B1b per-tier is **the arm §2 calls decisive**. The other three are here so
    the decisive number can be read in context: B1 says how much of B1b's
    performance is plain length, and the pooled variants say whether the
    tier interaction the per-tier fit spends degrees of freedom on is real.
    """
    return [
        SplitRegressionBaseline(ds.SURFACE_FEATURES, "B1_surface_pooled"),
        SplitRegressionBaseline(ds.BREADTH_FEATURES, "B1b_breadth_pooled"),
        SplitRegressionBaseline(ds.SURFACE_FEATURES, "B1_surface_pertier", per_tier=True),
        SplitRegressionBaseline(ds.BREADTH_FEATURES, "B1b_breadth_pertier", per_tier=True),
    ]


def surface_baseline(*, leave_one_out: bool = True) -> RegressionBaseline:
    """B1 — countable surface features only."""
    suffix = "" if leave_one_out else "_insample"
    return RegressionBaseline(
        ds.SURFACE_FEATURES, f"B1_surface{suffix}", leave_one_out=leave_one_out
    )


def breadth_baseline(*, leave_one_out: bool = True) -> RegressionBaseline:
    """B1b — B1 plus mechanically counted cross-document breadth."""
    suffix = "" if leave_one_out else "_insample"
    return RegressionBaseline(
        ds.BREADTH_FEATURES, f"B1b_breadth{suffix}", leave_one_out=leave_one_out
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FitReport:
    """A fitted baseline's coefficients, plus the in-sample/LOO honesty gap."""

    name: str
    model_cost: LinearModel
    model_quality: LinearModel
    sigma_cost_insample: float
    sigma_cost_loo: float

    @property
    def overfit_gap(self) -> float:
        return self.sigma_cost_loo - self.sigma_cost_insample


def fit_report(baseline: RegressionBaseline, data: ds.ProbeDataset) -> FitReport:
    """Fit on everything (for coefficients) and score both ways (for honesty)."""
    from agents.provider.estimation.metrics import sigma_cost

    features = {
        r.probe_id: ds.mechanical_features(r.visible) for r in data.requirements
    }
    cost_targets, quality_targets = baseline._targets(data)

    realized = {
        (r.probe_id, tier): r.realized_cost(tier) for r, tier in data.cells()
    }

    def score(leave_one_out: bool) -> float:
        predictions = baseline.predict(data, leave_one_out=leave_one_out)
        return sigma_cost(
            [p.predicted_cost_usd for p in predictions],
            [realized[(p.probe_id, p.tier)] for p in predictions],
        )

    return FitReport(
        name=baseline.name,
        model_cost=baseline._fit(features, cost_targets, None),
        model_quality=baseline._fit(features, quality_targets, None),
        sigma_cost_insample=score(False),
        sigma_cost_loo=score(True),
    )


def all_baselines() -> List[object]:
    """The offline arms, in the order the report tables them.

    Both the in-sample and the leave-one-out variant of B1/B1b are included, and
    the report needs both to be read honestly. They answer different questions:

    * **In-sample** is the *best case* for a mechanical predictor — what a
      reviewer means by "a 3-line regression would do this". It is the bar the
      LLM must clear to claim it is doing more than counting.
    * **LOO** is what the regression would actually deliver on a requirement it
      has not seen, which is the only thing a deployed agent could use.

    On v1 the two differ enormously (B1b: 0.1175 vs 0.2483), and reporting
    either alone would be misleading in opposite directions — the in-sample
    number overstates a baseline fit on ten points, the LOO number makes a
    control look easy to beat when the reason it fails is that ten points
    cannot support it.
    """
    return [
        TierMeanBaseline(),
        surface_baseline(leave_one_out=False),
        breadth_baseline(leave_one_out=False),
        surface_baseline(),
        breadth_baseline(),
    ]
