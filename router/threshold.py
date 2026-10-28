"""Choosing the routing threshold tau* (DESIGN.md §3.4; TASKS.md T21-T23).

Rule: send query i to the large model iff  score_i > tau.  `score` is whatever the router ranks by:
delta_hat (decision rule v1) or the cost-normalised gain G (rule v2).

T21  `sweep`             exact quality / cost / PGR at every breakpoint, O(n log n), no grid search.
     `upper_envelope`    the upper-left concave hull of the (cost, quality) cloud: the Pareto frontier
                         when randomising between two thresholds is allowed.
T22  `hoeffding_*`, `wsr_pvalues`, `bootstrap_*`
                         one-sided lower confidence bounds / p-values for "PGR(tau) >= rho".
T23  `select_tau`        Learn-then-Test fixed-sequence selection of tau* on the cal split only.
     `evaluate_at_tau`   step 5: apply a frozen tau to the test split once, with paired-bootstrap CIs.

The statistic behind T22/T23. With a PGR target rho (0.95), define per query

    e_i(tau) = s_{route(i)} - [(1 - rho) * s_S,i + rho * s_L,i]

Then mean_i e_i(tau) >= 0  <=>  Q(tau) >= Q_S + rho (Q_L - Q_S)  <=>  PGR(tau) >= rho.
`e` is paired (both arms are scored on the same query), so its variance is far smaller than the
variance of Q(tau) alone, and the bound also accounts for Q_S and Q_L being estimated from the same
cal sample rather than known. A query sent to L contributes (1 - rho)(s_L - s_S); one kept on S
contributes rho (s_S - s_L).

Validity notes, for the write-up (T52):
  * Fixed-sequence testing controls the family-wise error at delta without a Bonferroni penalty
    when the thresholds are fixed BEFORE looking at cal (Angelopoulos et al., Learn then Test). Build
    the grid from the train split (`threshold_grid`), not from cal. Using cal's own breakpoints is
    allowed by the API but is then a (mild) data-dependent grid; say so if you do it.
  * Three p-values, measured against each other in tests/test_threshold.py and notes/threshold_check.md:
      - "hoeffding": finite-sample valid, loose (certifies little at realistic n_cal).
      - "wsr": Waudby-Smith & Ramdas betting bound, the one RCPS (Bates et al. 2021) uses. Finite-
        sample valid AND adapts to the small variance of the paired statistic: about as tight as the
        bootstrap at n_cal >= 2,000. The DEFAULT.
      - "bootstrap": the paired bootstrap DESIGN.md §3.4 recommends. Asymptotic only: with a tiny
        gap (S 80%, L 84%) and n_cal = 2,000 it failed the target on fresh data in 15% of draws,
        three times delta. Report it, but do not rest the guarantee on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

THRESHOLD_VERSION = "threshold-1.0"


def _arr(x, n=None, fill=None) -> np.ndarray:
    if x is None:
        return np.full(n, fill, dtype=float)
    return np.asarray(x, dtype=float)


def _check(scores, s_S, s_L):
    scores, s_S, s_L = (np.asarray(v, dtype=float) for v in (scores, s_S, s_L))
    if not (scores.shape == s_S.shape == s_L.shape) or scores.ndim != 1:
        raise ValueError("scores, s_S and s_L must be 1-D arrays of the same length")
    if len(scores) == 0:
        raise ValueError("need at least one query")
    if np.isnan(scores).any():
        raise ValueError("scores contain NaN")
    return scores, s_S, s_L


# ----------------------------------------------------------------------------------------------------
# T21: exact sweep and envelope
# ----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Sweep:
    """Routing outcomes at a set of thresholds, ordered from most to least frontier calls."""

    tau: np.ndarray       # route to L iff score > tau
    k: np.ndarray         # queries routed to L
    alpha: np.ndarray     # k / n, the frontier-call rate
    quality: np.ndarray   # Q(tau)
    cost: np.ndarray      # mean cost per query C(tau)
    pgr: np.ndarray       # (Q - Q_S) / (Q_L - Q_S); NaN when there is no headroom
    q_S: float
    q_L: float
    c_all_L: float        # mean cost of always-frontier
    n: int

    def cpt(self, rho: float = 0.95) -> float:
        """CPT(rho): the smallest frontier-call rate whose PGR reaches rho (1.0 if none does)."""
        ok = self.pgr >= rho - 1e-12
        return float(self.alpha[ok].min()) if ok.any() else 1.0

    def savings(self, rho: float = 0.95) -> float:
        """1 - C / C_L at the cheapest threshold whose PGR reaches rho."""
        ok = self.pgr >= rho - 1e-12
        if not ok.any() or self.c_all_L <= 0:
            return 0.0
        return float(1.0 - self.cost[ok].min() / self.c_all_L)

    def apgr(self) -> float:
        """Area under PGR(alpha) on [0, 1], linear between breakpoints (= random tie-breaking)."""
        a, p = self.alpha, self.pgr
        order = np.argsort(a, kind="stable")
        a, p = a[order], p[order]
        if a[0] > 0:
            a, p = np.r_[0.0, a], np.r_[0.0, p]
        if a[-1] < 1:
            a, p = np.r_[a, 1.0], np.r_[p, 1.0]
        return float(np.sum((a[1:] - a[:-1]) * (p[1:] + p[:-1]) / 2.0))


class _Prefix:
    """Prefix sums over queries sorted by score, descending. Shared by sweep and the bounds."""

    def __init__(self, scores, *cols):
        self.order = np.argsort(-scores, kind="stable")
        self.sorted_scores = scores[self.order]
        self.n = len(scores)
        self.cums = [np.r_[0.0, np.cumsum(np.asarray(c, dtype=float)[self.order])] for c in cols]

    def k_for(self, taus: np.ndarray) -> np.ndarray:
        """Number of queries with score strictly greater than each tau."""
        asc = self.sorted_scores[::-1]
        return self.n - np.searchsorted(asc, taus, side="right")

    def routed(self, k: np.ndarray, top: int, rest: int) -> np.ndarray:
        """sum over the top-k queries of column `top`, plus sum over the others of column `rest`."""
        a, b = self.cums[top], self.cums[rest]
        return a[k] + (b[-1] - b[k])


def breakpoints(scores) -> np.ndarray:
    """Every threshold at which the routing changes: the distinct scores descending, then -inf.

    tau = a distinct score sends the queries strictly above it to L (so tau = max(score) sends
    nothing to L); tau = -inf sends everything. Between consecutive distinct scores nothing
    changes, so these are all the candidates there are. Returned order is most-to-S first; every
    consumer sorts into its own convention.
    """
    u = np.unique(np.asarray(scores, dtype=float))[::-1]
    return np.r_[u, -np.inf]


def threshold_grid(reference_scores, n: int = 1000) -> np.ndarray:
    """A threshold grid fixed in advance from OTHER data (the train split), for Learn-then-Test."""
    r = np.asarray(reference_scores, dtype=float)
    q = np.unique(np.quantile(r, np.linspace(0.0, 1.0, n + 1)))
    return np.r_[np.inf, q[::-1], -np.inf]


def evaluate_thresholds(scores, s_S, s_L, taus, c_S=None, c_L=None) -> Sweep:
    """Exact Q, C and PGR at the given thresholds (any order in, most-frontier-calls-first out)."""
    scores, s_S, s_L = _check(scores, s_S, s_L)
    n = len(scores)
    c_S = _arr(c_S, n, 0.0)
    c_L = _arr(c_L, n, 1.0)
    taus = np.unique(np.asarray(taus, dtype=float))          # ascending = most to L first
    pre = _Prefix(scores, s_L, s_S, c_L, c_S)
    k = pre.k_for(taus)
    quality = pre.routed(k, 0, 1) / n
    cost = pre.routed(k, 2, 3) / n
    q_S, q_L = float(s_S.mean()), float(s_L.mean())
    head = q_L - q_S
    pgr = (quality - q_S) / head if abs(head) > 1e-12 else np.full(len(taus), np.nan)
    return Sweep(tau=taus, k=k, alpha=k / n, quality=quality, cost=cost, pgr=pgr,
                 q_S=q_S, q_L=q_L, c_all_L=float(c_L.mean()), n=n)


def sweep(scores, s_S, s_L, c_S=None, c_L=None) -> Sweep:
    """T21: the exact routing curve at every breakpoint. O(n log n)."""
    return evaluate_thresholds(scores, s_S, s_L, breakpoints(scores), c_S=c_S, c_L=c_L)


def brute_force(scores, s_S, s_L, taus, c_S=None, c_L=None) -> tuple[np.ndarray, np.ndarray]:
    """O(n * m) reference implementation of (Q, C) at each tau. Used only to test `sweep`."""
    scores, s_S, s_L = _check(scores, s_S, s_L)
    c_S = _arr(c_S, len(scores), 0.0)
    c_L = _arr(c_L, len(scores), 1.0)
    q, c = [], []
    for t in taus:
        to_L = scores > t
        q.append(np.where(to_L, s_L, s_S).mean())
        c.append(np.where(to_L, c_L, c_S).mean())
    return np.array(q), np.array(c)


def upper_envelope(x, y) -> np.ndarray:
    """Indices of the upper-left concave envelope of the points (x = cost, y = quality).

    Starts at the cheapest point (best quality among ties) and ends at the highest-quality point
    (cheapest among ties). Any point strictly below the hull is dominated by a randomised mix of two
    hull points and is never optimal (DESIGN.md §3.4 step 2). Monotone-chain, O(m log m).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    order = np.lexsort((-y, x))                      # by x ascending, then y descending
    hull: list[int] = []
    best_y = -np.inf
    for i in order:
        if y[i] <= best_y + 1e-15:                    # not better than something cheaper: dominated
            continue
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            # drop b if it lies on or below the chord a -> i
            cross = (x[b] - x[a]) * (y[i] - y[a]) - (y[b] - y[a]) * (x[i] - x[a])
            if cross >= -1e-15:
                hull.pop()
            else:
                break
        hull.append(int(i))
        best_y = y[i]
    return np.array(hull, dtype=int)


def envelope_at(x_hull, y_hull, x_query) -> np.ndarray:
    """Envelope value at x (linear interpolation = randomising between two hull thresholds)."""
    return np.interp(np.asarray(x_query, dtype=float), x_hull, y_hull)


def envelope_cpt(sw: Sweep, rho: float = 0.95) -> float:
    """CPT(rho) on the (alpha, PGR) concave envelope: the frontier-call rate a randomised policy needs."""
    idx = upper_envelope(sw.alpha, sw.pgr)
    a, p = sw.alpha[idx], sw.pgr[idx]
    if p.max() < rho - 1e-12:
        return 1.0
    j = int(np.argmax(p >= rho - 1e-12))
    if j == 0:
        return float(a[0])
    # linear interpolation between hull points j-1 and j
    t = (rho - p[j - 1]) / (p[j] - p[j - 1])
    return float(a[j - 1] + t * (a[j] - a[j - 1]))


def reference_curves(s_S, s_L, c_S=None, c_L=None, seed: int = 0) -> dict[str, Sweep]:
    """Random-router and oracle-router sweeps, required on every Pareto plot (DESIGN.md §3.3).

    The random router's exact expected curve is the straight line PGR = alpha; the sweep returned
    here is one random draw (seeded), so it wobbles around that line as a real random router would.
    """
    s_S, s_L = np.asarray(s_S, dtype=float), np.asarray(s_L, dtype=float)
    rng = np.random.default_rng(seed)
    gap = s_L - s_S
    # Oracle: true gap, with cheaper escalations first among equal gaps when costs differ.
    c_L_ = _arr(c_L, len(gap), 1.0)
    oracle_score = gap - 1e-9 * (c_L_ / (np.abs(c_L_).max() + 1e-12)) + rng.uniform(0, 1e-12, len(gap))
    return {
        "random": sweep(rng.uniform(size=len(gap)), s_S, s_L, c_S, c_L),
        "oracle": sweep(oracle_score, s_S, s_L, c_S, c_L),
    }


# ----------------------------------------------------------------------------------------------------
# T22: lower confidence bounds
# ----------------------------------------------------------------------------------------------------

def hoeffding_lcb(mean: float | np.ndarray, n: int, delta: float = 0.05, width: float = 1.0):
    """Hoeffding one-sided LCB for a mean of n i.i.d. values in an interval of length `width`.

    The output type matches the input: a scalar mean gives a scalar, an array gives an array.
    """
    bound = np.asarray(mean) - width * np.sqrt(np.log(1.0 / delta) / (2.0 * n))
    return float(bound) if np.ndim(mean) == 0 else bound


def _margin_columns(s_S, s_L, rho):
    """Per-query contributions to e(tau): a_i if sent to L, b_i if kept on S."""
    a = (1.0 - rho) * (s_L - s_S)
    b = rho * (s_S - s_L)
    return a, b


def margin(scores, s_S, s_L, taus, rho: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Mean of e(tau) at each tau (>= 0 iff PGR(tau) >= rho), and the k for each tau."""
    scores, s_S, s_L = _check(scores, s_S, s_L)
    a, b = _margin_columns(s_S, s_L, rho)
    pre = _Prefix(scores, a, b)
    k = pre.k_for(np.asarray(taus, dtype=float))
    return pre.routed(k, 0, 1) / len(scores), k


def hoeffding_pvalues(scores, s_S, s_L, taus, rho: float = 0.95) -> np.ndarray:
    """Finite-sample p-values for H0: PGR(tau) < rho, one per tau.

    Hoeffding with per-query ranges: a query sent to L has e in [-(1-rho), 1-rho] (width 2(1-rho)),
    one kept on S has e in [-rho, rho] (width 2 rho). So thresholds that escalate almost everything
    get a tight bound, which is what lets fixed-sequence testing get started.
    """
    scores, s_S, s_L = _check(scores, s_S, s_L)
    n = len(scores)
    e_bar, k = margin(scores, s_S, s_L, taus, rho)
    sum_r2 = k * (2 * (1 - rho)) ** 2 + (n - k) * (2 * rho) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.exp(-2.0 * (n * e_bar) ** 2 / sum_r2)
    p = np.where(e_bar > 0, p, 1.0)
    return np.clip(np.nan_to_num(p, nan=1.0), 0.0, 1.0)


def wsr_pvalues(scores, s_S, s_L, taus, rho: float = 0.95, delta: float = 0.05, seed: int = 0,
                chunk: int = 256) -> np.ndarray:
    """Finite-sample p-values for H0: PGR(tau) < rho via the Waudby-Smith-Ramdas betting martingale.

    Maps e in [-rho, rho] to a loss l = (rho - e) / (2 rho) in [0, 1], so H0 is "mean loss >= 1/2".
    Bets with the predictable-plug-in sizes of WSR (2023) as used by RCPS (Bates et al. 2021, code
    `WSR_mu_plus`), over a fixed seeded permutation of the queries. By Ville's inequality
    p = 1 / max_t K_t is a valid p-value. `delta` only tunes the bet sizes, not validity.
    """
    scores, s_S, s_L = _check(scores, s_S, s_L)
    n = len(scores)
    perm = np.random.default_rng(seed).permutation(n)
    a, b = _margin_columns(s_S, s_L, rho)
    sc, a, b = scores[perm], a[perm], b[perm]
    taus = np.asarray(taus, dtype=float)
    t = np.arange(1, n + 1, dtype=float)
    out = np.empty(len(taus))
    for start in range(0, len(taus), chunk):
        tt = taus[start:start + chunk, None]
        e = np.where(sc[None, :] > tt, a[None, :], b[None, :])
        loss = (rho - e) / (2 * rho)
        mu = (np.cumsum(loss, axis=1) + 0.5) / (1 + t)
        s2 = (np.cumsum((loss - mu) ** 2, axis=1) + 0.25) / (1 + t)
        s2 = np.concatenate([np.full((len(tt), 1), 0.25), s2[:, :-1]], axis=1)
        nu = np.minimum(np.sqrt(2 * np.log(1 / delta) / (n * s2)), 1.0)
        log_k = np.cumsum(np.log1p(-nu * (loss - 0.5)), axis=1)
        out[start:start + chunk] = np.exp(-np.maximum(log_k.max(axis=1), 0.0))
    return np.clip(out, 0.0, 1.0)


def _bootstrap_margins(scores, s_S, s_L, taus, rho, B, seed, chunk=200) -> tuple[np.ndarray, np.ndarray]:
    """(B, m) bootstrap replicates of mean e(tau), resampling queries (pairs kept together)."""
    scores, s_S, s_L = _check(scores, s_S, s_L)
    n = len(scores)
    a, b = _margin_columns(s_S, s_L, rho)
    pre = _Prefix(scores, a, b)
    k = pre.k_for(np.asarray(taus, dtype=float))
    a_s, b_s = a[pre.order], b[pre.order]
    rng = np.random.default_rng(seed)
    out = np.empty((B, len(k)), dtype=np.float32)
    for start in range(0, B, chunk):
        m = min(chunk, B - start)
        w = rng.multinomial(n, np.full(n, 1.0 / n), size=m).astype(np.float64)   # (m, n) counts
        ca = np.concatenate([np.zeros((m, 1)), np.cumsum(w * a_s, axis=1)], axis=1)
        cb = np.concatenate([np.zeros((m, 1)), np.cumsum(w * b_s, axis=1)], axis=1)
        out[start:start + m] = (ca[:, k] + (cb[:, -1:] - cb[:, k])) / n
    return out, k


def bootstrap_lcb(scores, s_S, s_L, taus, rho=0.95, delta=0.05, B=2000, seed=0) -> np.ndarray:
    """Paired percentile-bootstrap (1 - delta) LCB of mean e(tau) at each tau. Feasible iff >= 0."""
    reps, _ = _bootstrap_margins(scores, s_S, s_L, taus, rho, B, seed)
    return np.quantile(reps, delta, axis=0)


def bootstrap_pvalues(scores, s_S, s_L, taus, rho=0.95, B=2000, seed=0) -> np.ndarray:
    """Bootstrap p-values for H0: PGR(tau) < rho: the share of replicates with mean e <= 0."""
    reps, _ = _bootstrap_margins(scores, s_S, s_L, taus, rho, B, seed)
    return (np.sum(reps <= 0, axis=0) + 1) / (B + 1)


# ----------------------------------------------------------------------------------------------------
# T23: Learn-then-Test selection, and the one-shot test report
# ----------------------------------------------------------------------------------------------------

@dataclass
class TauSelection:
    method: str
    rho: float
    delta: float
    tau_naive: float                 # largest tau whose point estimate meets the target
    tau_star: float                  # largest tau certified by fixed-sequence testing (= -inf if none)
    certified: bool                  # False -> nothing beyond always-frontier could be certified
    n_cal: int
    n_tested: int                    # thresholds tested before the first failure
    cal: dict = field(default_factory=dict)   # alpha / quality / pgr / cost on cal at tau_star
    naive: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in self.__dict__.items()}


def _row(sw: Sweep, i: int) -> dict:
    return {"tau": float(sw.tau[i]), "alpha": float(sw.alpha[i]), "quality": float(sw.quality[i]),
            "pgr": float(sw.pgr[i]), "cost": float(sw.cost[i])}


def select_tau(
    scores, s_S, s_L, *, taus=None, c_S=None, c_L=None, rho: float = 0.95, delta: float = 0.05,
    method: str = "wsr", B: int = 2000, seed: int = 0,
) -> TauSelection:
    """Choose tau* on the CAL split (DESIGN.md §3.4 steps 3-4). Call it with cal data only.

    Fixed-sequence (Learn-then-Test): walk the thresholds from the most conservative (everything to
    L) to the most aggressive, test H0: PGR(tau) < rho at level delta each, and stop at the first
    failure. tau* is the last threshold rejected. Always-frontier has PGR = 1 by definition, so it
    is accepted without a test.

    method: "wsr" (default: finite-sample valid and tight), "hoeffding" (valid, loose),
    "bootstrap" (DESIGN.md's paired bootstrap; asymptotic only), or "naive" (point estimate, no
    guarantee; reported to show the optimism gap).
    """
    if method not in ("wsr", "bootstrap", "hoeffding", "naive"):
        raise ValueError("method must be 'wsr', 'bootstrap', 'hoeffding' or 'naive'")
    scores, s_S, s_L = _check(scores, s_S, s_L)
    taus = breakpoints(scores) if taus is None else np.r_[np.asarray(taus, dtype=float), -np.inf]
    sw = evaluate_thresholds(scores, s_S, s_L, taus, c_S=c_S, c_L=c_L)   # ascending tau
    e_bar, _ = margin(scores, s_S, s_L, sw.tau, rho)

    point_ok = e_bar >= -1e-12
    i_naive = int(np.flatnonzero(point_ok).max()) if point_ok.any() else 0

    if method == "naive":
        p = np.where(point_ok, 0.0, 1.0)
    elif method == "hoeffding":
        p = hoeffding_pvalues(scores, s_S, s_L, sw.tau, rho)
    elif method == "wsr":
        p = wsr_pvalues(scores, s_S, s_L, sw.tau, rho, delta=delta, seed=seed)
    else:
        p = bootstrap_pvalues(scores, s_S, s_L, sw.tau, rho, B=B, seed=seed)
    p = np.where(sw.k == sw.n, 0.0, p)            # always-frontier: PGR = 1, no test needed

    passed = p <= delta
    fails = np.flatnonzero(~passed)
    last = (fails[0] - 1) if len(fails) else len(passed) - 1   # >= 0: tau = -inf always passes
    return TauSelection(
        method=method, rho=rho, delta=delta,
        tau_naive=float(sw.tau[i_naive]), tau_star=float(sw.tau[last]),
        certified=bool(sw.k[last] < sw.n), n_cal=sw.n,
        n_tested=int(fails[0] + 1) if len(fails) else len(passed),
        cal=_row(sw, last), naive=_row(sw, i_naive),
    )


def evaluate_at_tau(scores, s_S, s_L, tau: float, *, c_S=None, c_L=None, B: int = 2000,
                    seed: int = 0, level: float = 0.95) -> dict:
    """Step 5: apply a frozen tau to the TEST split once. Point values plus paired-bootstrap CIs."""
    scores, s_S, s_L = _check(scores, s_S, s_L)
    n = len(scores)
    c_S = _arr(c_S, n, 0.0)
    c_L = _arr(c_L, n, 1.0)
    to_L = scores > tau
    s_r = np.where(to_L, s_L, s_S)
    c_r = np.where(to_L, c_L, c_S)

    def stats(idx):
        q, qs, ql = s_r[idx].mean(), s_S[idx].mean(), s_L[idx].mean()
        cl = c_L[idx].mean()
        head = ql - qs
        return {
            "alpha": to_L[idx].mean(),
            "quality": q,
            "q_S": qs,
            "q_L": ql,
            "pgr": (q - qs) / head if abs(head) > 1e-12 else np.nan,
            "cost": c_r[idx].mean(),
            "savings": 1.0 - c_r[idx].mean() / cl if cl > 0 else np.nan,
        }

    point = stats(np.arange(n))
    rng = np.random.default_rng(seed)
    reps = [stats(rng.integers(0, n, n)) for _ in range(B)]
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    ci = {k: (float(np.nanquantile([r[k] for r in reps], lo)), float(np.nanquantile([r[k] for r in reps], hi)))
          for k in point}
    return {"tau": float(tau), "n": n, **{k: float(v) for k, v in point.items()}, "ci": ci, "ci_level": level}
