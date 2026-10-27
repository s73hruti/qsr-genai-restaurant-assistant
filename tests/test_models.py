from datetime import date

from qsr_assistant.monitoring.drift import psi


def test_forecaster_beats_seasonal_naive(registry):
    m = registry.metrics["forecaster"]
    assert m["mape"] < m["mape_naive_baseline"] and m["mape"] < 0.10


def test_inventory_plan_has_safety_stock(registry):
    plan = registry.forecaster.inventory_plan(1, date(2026, 1, 5), 3)
    assert (plan.recommended_order >= plan.expected_need).all()
    assert (plan.safety_stock > 0).all() and len(plan) == 18


def test_wait_model_beats_baseline_and_p90_above_mean(registry):
    m = registry.metrics["waittime"]
    assert m["mae_min"] < m["mae_baseline_min"] / 2
    busy = registry.waittime.predict(1, 12, 4, queue_len=15)
    quiet = registry.waittime.predict(1, 12, 4, queue_len=2)
    assert busy["p90_min"] >= busy["wait_min"] and busy["wait_min"] > quiet["wait_min"]


def test_recommender_context_changes_ranking(registry):
    hot = [r["name"] for r in registry.recommender.recommend(None, 14, 36, k=4)]
    cold = [r["name"] for r in registry.recommender.recommend(None, 14, 12, k=4)]
    assert hot != cold
    assert "Hot Coffee" in [r["name"] for r in registry.recommender.recommend(None, 8, 12, k=5)]


def test_cross_sell_pairs_burger_with_fries_or_cola(registry):
    names = [r["name"] for r in registry.recommender.cross_sell([1], k=3)]
    assert {"Large Fries", "Cola", "Medium Fries"} & set(names)


def test_psi_zero_for_identical_and_high_for_shift():
    import numpy as np
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, a) < 0.01 and psi(a, a + 2) > 0.25
