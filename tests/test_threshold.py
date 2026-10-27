"""T21-T23: exact sweep, envelope, lower bounds and Learn-then-Test (router/threshold.py)."""
import numpy as np
import pytest

from router import threshold as th


def _case(rng, n, ties=False, soft=False):
    scores = rng.integers(0, 6, n).astype(float) if ties else rng.normal(size=n)
    if soft:
        s_S, s_L = rng.uniform(size=n), rng.uniform(size=n)
    else:
        s_S, s_L = rng.integers(0, 2, n).astype(float), rng.integers(0, 2, n).astype(float)
    c_L = rng.uniform(0.001, 0.02, n)
    return scores, s_S, s_L, c_L


@pytest.mark.parametrize("case", range(200))
def test_sweep_matches_brute_force_on_200_random_cases(case):
    rng = np.random.default_rng(case)
    scores, s_S, s_L, c_L = _case(rng, int(rng.integers(1, 60)), ties=case % 3 == 0, soft=case % 2 == 0)
    sw = th.sweep(scores, s_S, s_L, c_L=c_L)
    q, c = th.brute_force(scores, s_S, s_L, sw.tau, c_L=c_L)
    np.testing.assert_allclose(sw.quality, q, atol=1e-12)
    np.testing.assert_allclose(sw.cost, c, atol=1e-12)


def test_sweep_endpoints_and_order():
    rng = np.random.default_rng(1)
    scores, s_S, s_L, c_L = _case(rng, 50)
    sw = th.sweep(scores, s_S, s_L, c_L=c_L)
    assert sw.k[0] == 50 and sw.k[-1] == 0            # most frontier calls first
    assert np.all(np.diff(sw.k) < 0)                   # one breakpoint per distinct score
    assert sw.quality[0] == pytest.approx(s_L.mean()) and sw.quality[-1] == pytest.approx(s_S.mean())
    assert sw.pgr[0] == pytest.approx(1.0) and sw.pgr[-1] == pytest.approx(0.0)


def test_ties_move_together():
    sw = th.sweep([1, 1, 0], [0, 0, 0], [1, 1, 1])
    assert list(sw.k) == [3, 2, 0]                     # the two tied queries are never split


def test_cpt_apgr_on_a_perfect_router():
    # half the queries gain from L; a perfect router reaches PGR 1 at alpha 0.5
    s_S = np.array([0, 0, 1, 1.0]); s_L = np.array([1, 1, 1, 1.0])
    sw = th.sweep([4, 3, 2, 1], s_S, s_L)
    assert sw.cpt(0.95) == pytest.approx(0.5)
    assert sw.apgr() == pytest.approx(0.75)            # PGR rises 0 -> 1 over [0, .5], then flat
    assert sw.savings(0.95) == pytest.approx(0.5)


def test_random_and_oracle_bracket_a_real_router():
    rng = np.random.default_rng(3)
    n = 4000
    gap = rng.uniform(size=n) < 0.3
    s_L = np.ones(n); s_S = np.where(gap, 0.0, 1.0)
    router = gap + rng.normal(scale=0.8, size=n)        # noisy but informative
    sw = th.sweep(router, s_S, s_L)
    ref = th.reference_curves(s_S, s_L)
    assert ref["oracle"].apgr() >= sw.apgr() >= ref["random"].apgr()
    assert ref["random"].apgr() == pytest.approx(0.5, abs=0.03)


def test_upper_envelope_is_concave_and_dominates():
    rng = np.random.default_rng(4)
    x, y = rng.uniform(size=300), rng.uniform(size=300)
    idx = th.upper_envelope(x, y)
    hx, hy = x[idx], y[idx]
    assert np.all(np.diff(hx) > 0) and np.all(np.diff(hy) > 0)
    slopes = np.diff(hy) / np.diff(hx)
    assert np.all(np.diff(slopes) <= 1e-9)             # concave
    # every point lies on or below the envelope over the envelope's x-range
    inside = (x >= hx[0]) & (x <= hx[-1])
    assert np.all(y[inside] <= th.envelope_at(hx, hy, x[inside]) + 1e-12)
    assert idx[0] == np.lexsort((-y, x))[0]            # starts at the cheapest point


def test_envelope_cpt_never_worse_than_plain_cpt():
    rng = np.random.default_rng(5)
    scores, s_S, s_L, _ = _case(rng, 500)
    sw = th.sweep(scores, s_S, s_L)
    assert th.envelope_cpt(sw) <= sw.cpt() + 1e-12


def test_hoeffding_slack_matches_design_doc():
    # DESIGN.md §3.4: n = 1500, delta = .05 -> slack about 0.032
    assert 0.5 - th.hoeffding_lcb(0.5, 1500, 0.05) == pytest.approx(0.0316, abs=5e-4)


def test_margin_sign_is_pgr_target():
    rng = np.random.default_rng(6)
    scores, s_S, s_L, _ = _case(rng, 300)
    sw = th.sweep(scores, s_S, s_L)
    e, _ = th.margin(scores, s_S, s_L, sw.tau, rho=0.95)
    head = sw.q_L - sw.q_S
    np.testing.assert_allclose(e, sw.quality - (sw.q_S + 0.95 * head), atol=1e-12)


def _low_headroom(n, seed):
    """S is right 80% of the time, L 84%: a 4-point gap, the case where loose bounds certify nothing."""
    rng = np.random.default_rng(seed)
    s_S = (rng.uniform(size=n) < 0.80).astype(float)
    s_L = np.where(s_S == 1, 1.0, (rng.uniform(size=n) < 0.2).astype(float))
    score = (s_L - s_S) + rng.normal(scale=0.5, size=n)
    return score, s_S, s_L


def test_bounds_are_conservative_and_bootstrap_is_tighter():
    score, s_S, s_L = _low_headroom(3000, 7)
    taus = th.threshold_grid(_low_headroom(3000, 8)[0], n=200)
    e, _ = th.margin(score, s_S, s_L, taus)
    lcb = th.bootstrap_lcb(score, s_S, s_L, taus, B=500)
    assert np.all(lcb <= e + 1e-6)
    p_h = th.hoeffding_pvalues(score, s_S, s_L, taus)
    p_b = th.bootstrap_pvalues(score, s_S, s_L, taus, B=500)
    feasible = e > 0
    assert np.median(p_b[feasible]) <= np.median(p_h[feasible])


def test_tau_star_is_chosen_on_cal_and_is_no_more_aggressive_than_naive():
    score, s_S, s_L = _low_headroom(3000, 9)
    grid = th.threshold_grid(_low_headroom(3000, 10)[0], n=300)       # fixed from "train"
    naive = th.select_tau(score, s_S, s_L, taus=grid, method="naive")
    boot = th.select_tau(score, s_S, s_L, taus=grid, method="bootstrap", B=500)
    wsr = th.select_tau(score, s_S, s_L, taus=grid, method="wsr")
    hoef = th.select_tau(score, s_S, s_L, taus=grid, method="hoeffding")
    assert naive.cal["pgr"] >= 0.95 - 1e-9
    for sel in (boot, wsr, hoef):                                       # conservative = more L calls
        assert sel.cal["alpha"] >= naive.naive["alpha"] - 1e-12
    # a 4-point gap with n_cal = 3,000: the finite-sample bounds honestly certify nothing
    assert not hoef.certified and hoef.cal["alpha"] == 1.0


def _moderate(n, seed):
    """S 60%, L 85%: a realistic MMLU-sized gap."""
    rng = np.random.default_rng(seed)
    s_S = (rng.uniform(size=n) < 0.6).astype(float)
    s_L = np.where(s_S == 1, (rng.uniform(size=n) < 0.95).astype(float),
                   (rng.uniform(size=n) < (0.85 - 0.95 * 0.6) / 0.4).astype(float))
    return (s_L - s_S) + rng.normal(scale=0.7, size=n), s_S, s_L


@pytest.mark.parametrize("method", ["wsr", "hoeffding"])
def test_learn_then_test_controls_risk_on_fresh_data(method):
    """Over repeated cal draws, tau* misses the target on fresh data at most ~delta of the time,
    while the naive point-estimate choice misses it far more often."""
    grid = th.threshold_grid(_moderate(5000, 999)[0], n=100)
    population = _moderate(200_000, 1)
    fails = naive_fails = certified = 0
    trials = 30
    for t in range(trials):
        cal = _moderate(2000, 100 + t)
        sel = th.select_tau(*cal, taus=grid, method=method)
        nav = th.select_tau(*cal, taus=grid, method="naive")
        fails += th.margin(*population, [sel.tau_star])[0][0] < 0
        naive_fails += th.margin(*population, [nav.tau_star])[0][0] < 0
        certified += sel.certified
    assert fails / trials <= 0.10
    assert naive_fails / trials >= 0.2
    assert certified == trials


def test_wsr_is_nearly_as_tight_as_bootstrap_and_tighter_than_hoeffding():
    grid = th.threshold_grid(_moderate(5000, 999)[0], n=100)
    cal = _moderate(4000, 3)
    a = {m: th.select_tau(*cal, taus=grid, method=m, B=400).cal["alpha"] for m in ("wsr", "bootstrap", "hoeffding")}
    assert a["wsr"] <= a["hoeffding"] - 0.03
    assert a["wsr"] <= a["bootstrap"] + 0.05


def test_evaluate_at_tau_reports_point_and_ci():
    score, s_S, s_L = _low_headroom(2000, 11)
    rep = th.evaluate_at_tau(score, s_S, s_L, 0.0, B=300)
    assert rep["ci"]["pgr"][0] <= rep["pgr"] <= rep["ci"]["pgr"][1]
    assert 0 <= rep["alpha"] <= 1 and rep["savings"] == pytest.approx(1 - rep["alpha"])


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        th.sweep([1, 2], [0], [1, 1])
    with pytest.raises(ValueError):
        th.select_tau([1.0], [0.0], [1.0], method="magic")

def test_hoeffding_lcb_scalar_stays_scalar():
    """A scalar mean must return a scalar bound (it used to return a 0-d array)."""
    from router.threshold import hoeffding_lcb
    out = hoeffding_lcb(0.9, 1000)
    assert isinstance(out, float)
    assert hoeffding_lcb(0.9, 10000) > hoeffding_lcb(0.9, 1000)    # more data -> tighter bound
    arr = hoeffding_lcb(np.array([0.9, 0.9]), 1000)
    assert arr.shape == (2,)
