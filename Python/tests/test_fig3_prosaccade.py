"""Correctness checks for the Fig 3 prosaccade figure script.

Three kinds of test live here:

1. Pure-logic tests for the helpers that decide what gets drawn
   (``_window_is_informative``) and for the Wilson-interval reconstruction
   in the accuracy-vs-latency panels. These need no data and no cache.

2. A test of the quiver recolouring in :func:`_draw_quiver_panel`. That is
   the one place the figure reaches past a public API -- it repaints
   ``ax.collections[-1]`` in place because
   ``eyehead.analysis._draw_quiver_arrows`` hardcodes its colors and takes
   no color argument. If matplotlib ever changes what lands in
   ``ax.collections``, or if the helper starts adding artists of its own,
   the arrows would silently take the wrong colors and Panel B would
   misreport which saccades were correct. Hence a direct readback test.

3. Analysis tests that re-derive the numbers the figure plots straight from
   the cache, by a different route than the plotting code uses, and check
   the two agree. These skip cleanly when no population cache is present.

Runnable either way::

    pytest Python/tests/test_fig3_prosaccade.py
    python Python/tests/test_fig3_prosaccade.py

The second form exists because pytest is not currently installed in the
EyeHeadCoupling environment. ``unittest.SkipTest`` is used for skips since
pytest reports it as a skip and the built-in runner below understands it too.
"""
from pathlib import Path
from unittest import SkipTest
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # never open a window from a test
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis import Fig3_prosaccade as fig3
from analysis import prosaccade_session


# ---------------------------------------------------------------------------
# 1. Pure logic
# ---------------------------------------------------------------------------

def test_window_covering_whole_panel_is_not_drawn() -> None:
    """A window from <=0 to >=x_hi marks nothing but tints everything under
    it, which is what turned two panels brown. It must be suppressed."""
    assert fig3._window_is_informative((0.0, 1.1), 0.95) is False
    assert fig3._window_is_informative((0.0, 0.95), 0.95) is False


def test_partial_window_is_drawn() -> None:
    assert fig3._window_is_informative((0.2, 1.0), 0.95) is True
    assert fig3._window_is_informative((0.0, 0.5), 0.95) is True


def test_degenerate_window_inputs_are_not_drawn() -> None:
    assert fig3._window_is_informative(None, 0.95) is False
    assert fig3._window_is_informative((0.2, 1.0), float("nan")) is False


def test_accuracy_window_lower_edge_excludes_anticipatory_saccades() -> None:
    """The 0.2 s lower edge is the whole point of the windowed measure --
    guard against it being widened back to 0 without anyone noticing."""
    lo, hi = fig3.ACCURACY_WINDOW
    assert lo > 0.0, "accuracy window must exclude the earliest saccades"
    assert hi > lo


def test_wilson_reconstruction_from_fraction_is_lossless() -> None:
    """The accuracy panels recover a success count as ``round(frac * n)``
    because fraction_toward_target_by_latency returns only the fraction.
    That round-trip has to be exact, or the plotted intervals belong to a
    different count than the plotted point."""
    rng = np.random.default_rng(0)
    latencies = rng.uniform(0.0, 1.0, size=400)
    congruent = rng.random(400) < 0.7

    centers, frac, n_per_window = prosaccade_session.fraction_toward_target_by_latency(
        latencies, congruent, window_span=(0, 1.0),
    )
    checked = 0
    for i in np.flatnonzero(n_per_window > 0):
        n_i = int(n_per_window[i])
        k_i = int(round(frac[i] * n_i))
        assert abs(k_i / n_i - frac[i]) < 1e-12, (
            f"window {i}: round-trip changed the fraction "
            f"({k_i}/{n_i} != {frac[i]})"
        )
        checked += 1
    assert checked > 0, "no populated windows -- test proved nothing"


def test_wilson_interval_brackets_its_estimate() -> None:
    """At k=0 (and k=n) the Wilson bound is analytically exactly 0 (or 1),
    but the closed form evaluates to within ~1e-17 of it rather than on it --
    wilson_ci(0, 10) returns a lower bound of about -2.8e-17. Harmless: the
    accuracy panels clip to [0, 1] and the validity errorbars already clamp
    with max(0.0, ...). So allow float slack at the boundaries rather than
    demanding exact containment."""
    tol = 1e-9
    for k, n in ((0, 10), (5, 10), (10, 10), (1, 3), (97, 100)):
        lo, hi = prosaccade_session.wilson_ci(k, n)
        phat = k / n
        assert lo - tol <= phat <= hi + tol, f"wilson_ci({k}, {n}) = ({lo}, {hi})"
        assert -tol <= lo and hi <= 1.0 + tol, f"wilson_ci({k}, {n}) = ({lo}, {hi})"


# ---------------------------------------------------------------------------
# 2. The quiver recolouring hack
# ---------------------------------------------------------------------------

def test_quiver_arrows_recoloured_per_trial_congruency() -> None:
    """Each arrow must take the palette color matching its own congruency
    flag, in order. This is a readback of the actual drawn artist, not of
    the inputs."""
    congruent = np.array([True, False, True, False, True])
    n = congruent.size
    qd = {
        "x": np.linspace(-5, 5, n),
        "y": np.zeros(n),
        "dx": np.ones(n),
        "dy": np.zeros(n),
        "congruent": congruent,
    }

    fig, ax = plt.subplots()
    try:
        fig3._draw_quiver_panel(ax, qd, "Left")
        quivers = [c for c in ax.collections if type(c).__name__ == "Quiver"]
        assert len(quivers) == 1, (
            f"expected exactly one Quiver artist, found {len(quivers)} -- "
            f"the recolour targets ax.collections[-1] and would repaint the "
            f"wrong artist"
        )
        facecolors = np.asarray(quivers[0].get_facecolor())
        assert facecolors.shape[0] == n, (
            f"expected one color per arrow, got {facecolors.shape[0]} for {n} arrows"
        )

        expected = np.array([
            to_rgba(fig3.CORRECT_COLOR if c else fig3.INCORRECT_COLOR)
            for c in congruent
        ])
        # Compare RGB only: _draw_quiver_arrows sets alpha=0.5 on the artist,
        # which is intended and independent of the congruency colouring.
        np.testing.assert_allclose(facecolors[:, :3], expected[:, :3], atol=1e-6)
    finally:
        plt.close(fig)


def test_quiver_palette_is_not_the_upstream_green_red() -> None:
    """The recolour exists so Panel B stops speaking green/red. If the
    constants drift back, the accessibility fix is silently undone."""
    for name in ("CORRECT_COLOR", "INCORRECT_COLOR"):
        value = getattr(fig3, name)
        assert value not in ("tab:green", "tab:red", "green", "red"), (
            f"{name} is back to a red/green value ({value!r})"
        )


# ---------------------------------------------------------------------------
# 3. Analysis, checked against the cache
# ---------------------------------------------------------------------------

def _load_cache_or_skip() -> dict:
    try:
        return fig3._load_population_cache(fig3.EXPERIMENT_TYPE)
    except FileNotFoundError as exc:
        raise SkipTest(f"no population cache available: {exc}")


def test_pooled_trial_counts_equal_sum_over_sessions() -> None:
    """Per-animal pooling must neither drop nor double-count a session."""
    cache = _load_cache_or_skip()
    session_results = cache["session_results"]
    for animal, pooled in cache["animal_pooled"].items():
        rows = [r for r in cache["session_validity"] if r["animal_name"] == animal]
        assert rows, f"{animal} has no sessions in session_validity"

        expected_total = sum(r["n_total"] for r in rows)
        assert pooled["latency_outcome"]["n_total"] == expected_total, (
            f"{animal}: pooled n_total {pooled['latency_outcome']['n_total']} "
            f"!= sum over sessions {expected_total}"
        )

        expected_valid = sum(
            len(session_results[r["session_id"]]["latency_outcome"]["latencies"])
            for r in rows
        )
        assert len(pooled["latency_outcome"]["latencies"]) == expected_valid, (
            f"{animal}: pooled latency count != sum over sessions"
        )


def test_windowed_accuracy_matches_direct_recomputation() -> None:
    """The number Panel D's second column reports, re-derived by hand from
    the raw arrays rather than through congruency_in_window."""
    cache = _load_cache_or_skip()
    lo_edge, hi_edge = fig3.ACCURACY_WINDOW
    for animal, pooled in cache["animal_pooled"].items():
        lo = pooled["latency_outcome"]
        latencies = np.asarray(lo["latencies"])
        congruent = np.asarray(lo["congruent"])

        frac, n, ci_lo, ci_hi = prosaccade_session.congruency_in_window(
            latencies, congruent, window=fig3.ACCURACY_WINDOW,
        )
        # congruency_in_window selects [lo, hi) -- mirror that exactly.
        sel = (latencies >= lo_edge) & (latencies < hi_edge)
        assert n == int(np.count_nonzero(sel)), f"{animal}: window n mismatch"
        assert abs(frac - float(np.mean(congruent[sel]))) < 1e-12, (
            f"{animal}: windowed accuracy mismatch"
        )
        assert ci_lo <= frac <= ci_hi, f"{animal}: CI does not bracket estimate"


def test_windowed_accuracy_differs_from_all_trial_accuracy() -> None:
    """The windowed column replaced the all-trials one on the grounds that
    it measures something different. If the two ever coincide, the window is
    no longer excluding anything and the panel has quietly lost its point."""
    cache = _load_cache_or_skip()
    for animal, pooled in cache["animal_pooled"].items():
        lo = pooled["latency_outcome"]
        congruent = np.asarray(lo["congruent"])
        all_trials = float(np.mean(congruent))
        windowed, _, _, _ = prosaccade_session.congruency_in_window(
            np.asarray(lo["latencies"]), congruent, window=fig3.ACCURACY_WINDOW,
        )
        assert windowed > all_trials, (
            f"{animal}: windowed accuracy ({windowed:.3f}) is not above "
            f"all-trials accuracy ({all_trials:.3f}) -- the 0.2 s cut is "
            f"no longer removing chance-level early saccades"
        )


def test_precue_fraction_matches_its_own_array() -> None:
    cache = _load_cache_or_skip()
    for animal, pooled in cache["animal_pooled"].items():
        precue = np.asarray(pooled["precue_congruent"])
        if precue.size == 0:
            raise SkipTest(f"{animal} has no pre-cue trials")
        frac = float(np.mean(precue))
        assert 0.0 <= frac <= 1.0
        lo, hi = prosaccade_session.wilson_ci(int(np.sum(precue)), precue.size)
        assert lo <= frac <= hi


def test_supplement_per_animal_counts_sum_to_pooled_total() -> None:
    """The supplement splits by animal; the split must be exhaustive and
    disjoint with respect to the main figure's pooled panel."""
    cache = _load_cache_or_skip()
    animal_pooled = cache["animal_pooled"]
    if len(animal_pooled) < 2:
        raise SkipTest("only one animal cached -- nothing to split")

    from analysis import prosaccade_population as pp
    pooled_all = pp.pool_animal_sessions(list(animal_pooled.values()))

    per_animal = sum(
        len(p["latency_outcome"]["latencies"]) for p in animal_pooled.values()
    )
    assert len(pooled_all["latency_outcome"]["latencies"]) == per_animal, (
        "sum of per-animal trial counts != main figure's pooled count"
    )


def test_supplement_figure_builds() -> None:
    """Smoke test: one row of three panels per animal, and no exceptions."""
    cache = _load_cache_or_skip()
    n_animals = len(cache["animal_pooled"])
    fig = fig3.build_supplement_figure(fig3.EXPERIMENT_TYPE)
    try:
        assert len(fig.axes) == 3 * n_animals, (
            f"expected {3 * n_animals} axes for {n_animals} animals, "
            f"got {len(fig.axes)}"
        )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Minimal runner, so this file works without pytest installed
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import traceback

    tests = [
        (name, obj) for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    passed = skipped = failed = 0
    for name, fn in tests:
        try:
            fn()
        except SkipTest as exc:
            print(f"SKIP {name}: {exc}")
            skipped += 1
        except Exception:
            print(f"FAIL {name}")
            traceback.print_exc()
            failed += 1
        else:
            print(f"PASS {name}")
            passed += 1

    print(f"\n{passed} passed, {skipped} skipped, {failed} failed "
          f"(of {len(tests)})")
    sys.exit(1 if failed else 0)
