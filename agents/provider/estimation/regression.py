"""Regression's fit: a ridge regression of log cost on countable request features."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from agents.provider.estimation.llm_client import COST_MULTIPLIER_BOUNDS
from agents.provider.estimation.requirement import SURFACE_FEATURES

#: Numerical insurance on the normal equations, far too small to change a
#: well-posed fit.
RIDGE_EPSILON = 1e-8

#: Observations required per fitted parameter before the regression is used.
#: With 3 features the fit has 4 parameters, so it needs n >= 8.
GUARD_OBSERVATIONS_PER_PARAMETER = 2

#: Ridge shrinkage on standardised columns.
GUARD_RIDGE_LAMBDA = 1.0


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


@dataclass(frozen=True)
class RegressionEstimate:
    """Arm R's prediction, with the diagnostics that say whether to believe it."""

    predicted_cost: float
    multiplier: float
    n_observations: int
    usable_features: Tuple[str, ...]
    dropped_features: Tuple[str, ...]
    #: True when `R_guarded` declined to fit and returned the tier level. Not an
    #: error — the arm behaving as designed at small n (S7 §B.2).
    guard_withheld: bool = False

    @property
    def degenerate(self) -> bool:
        """True when the fit had no usable feature, so R collapsed to B0.

        Not a failure — it is P2's prediction made concrete. With k = 3 an agent
        cannot fit three coefficients on three points in any meaningful sense,
        and the honest baseline in the cold-start regime *is* the tier mean.
        """
        return not self.usable_features

    @property
    def exactly_determined(self) -> bool:
        """As many usable features as observations — interpolates, generalises poorly.

        Distinct from :attr:`degenerate`: the fit succeeds and returns a
        confident number built on no residual degrees of freedom.
        """
        return bool(self.usable_features) and len(self.usable_features) >= self.n_observations


def fit_cost_multiplier(
    observed: Sequence[Tuple[Dict[str, float], float]],
    target_features: Dict[str, float],
    tier_level_cost: float,
    *,
    feature_names: Sequence[str] = SURFACE_FEATURES,
    guard: bool = False,
) -> RegressionEstimate:
    """Arm **R**'s fit, on features alone. The one implementation of it.

    ``guard`` selects between S7 §B.2's two arms, which differ *only* here:

    * ``guard=False`` — **`R_naive`**. Always fits, however little it has. This
      is the arm S6 measured at sigma 0.9327 on 3 jobs, worse than guessing, and
      reproducing that curve is why it still runs.
    * ``guard=True`` — **`R_guarded`**. Ridge-penalised, and it declines to fit
      at all below :data:`GUARD_OBSERVATIONS_PER_PARAMETER` observations per
      parameter, returning the tier level instead. It therefore *is* the pure
      heuristic at cold start, by construction.

    The second is the honest opponent: if the hybrid only beat `R_naive`, the
    cold-start result would be an artifact of a strawman nobody would deploy.

    ``observed`` is ``[(features, realized_cost), ...]`` for the agent's own past
    jobs at one tier. Callers differ only in where those come from: the probe
    draws them (:func:`regression_estimate`), and the simulator's cold-start arm
    accumulates them from the jobs it actually won. **Both must fit identically**
    or "the regression an agent can construct" means two different things in the
    two places S6 compares.

    Target is ``log(realized / tier level)``, the same as S5's B1, so the
    multiplier is comparable to the LLM's; and :func:`baselines.fit_linear` is the
    same, so zero-variance columns are screened rather than producing arbitrary
    coefficients. With no usable observation the arm *is* the tier mean, which is
    P2's cold-start prediction rather than a failure — see
    :attr:`RegressionEstimate.degenerate`.
    """
    usable = [(row, cost) for row, cost in observed if cost > 0]
    if not usable or tier_level_cost <= 0:
        return RegressionEstimate(
            predicted_cost=tier_level_cost, multiplier=1.0, n_observations=0,
            usable_features=(), dropped_features=tuple(feature_names),
        )

    rows = [row for row, _ in usable]
    targets = [math.log(cost / tier_level_cost) for _, cost in usable]

    if guard:
        # Count the parameters this fit would actually estimate — features that
        # survive the zero-variance screen, plus the intercept — and refuse if
        # the history cannot support them. Screening first matters: on a history
        # where two columns are constant, the fit is univariate and needs 4
        # observations, not 8.
        informative = [
            name for name in feature_names
            if len({row.get(name) for row in rows}) > 1
        ]
        required = GUARD_OBSERVATIONS_PER_PARAMETER * (len(informative) + 1)
        if len(usable) < required:
            return RegressionEstimate(
                predicted_cost=tier_level_cost,
                multiplier=1.0,
                n_observations=len(usable),
                usable_features=(),
                dropped_features=tuple(feature_names),
                guard_withheld=True,
            )

    model = fit_linear(
        rows, targets, feature_names,
        ridge_lambda=GUARD_RIDGE_LAMBDA if guard else 0.0,
    )

    low, high = COST_MULTIPLIER_BOUNDS
    # Same bounds the LLM's output is held to. An exactly-determined fit
    # extrapolates hard, and an unbounded R would lose to the LLM on outliers
    # the LLM is protected from.
    #
    # **Clamped in log space, before exponentiating.** Identical arithmetic
    # wherever ``exp`` is finite, and the only version that survives where it is
    # not: three near-collinear observations can put ``predict`` in the hundreds,
    # and ``exp(800)`` raises ``OverflowError`` before any bound can be applied.
    # Found by S6 §E's cold-start arm, which fits on whatever an agent happened to
    # win rather than on curated draws — but the defect was always here, and the
    # offline probe simply never fed it an extreme enough history.
    log_multiplier = model.predict(target_features)
    if log_multiplier != log_multiplier:            # NaN from a degenerate solve
        log_multiplier = 0.0
    multiplier = math.exp(
        max(math.log(low), min(math.log(high), log_multiplier))
    )

    return RegressionEstimate(
        predicted_cost=tier_level_cost * multiplier,
        multiplier=multiplier,
        n_observations=len(usable),
        usable_features=tuple(model.coefficients),
        dropped_features=tuple(model.dropped),
    )
