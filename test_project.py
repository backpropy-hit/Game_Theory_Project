"""Run with:  python -m pytest -q
Checks that our code reproduces the paper's numbers, plus our extension's main claims."""
import pytest
from paper_game import (payoff_table, pure_nash, mixed_nash_2x2, bayesian_table, bayesian_nash,
                        coherent_probability, S_ACTIONS, A_ACTIONS, S_PLANS, A_PLANS)
from defect_game import arm_choice, solve, PARAMS


def test_table_III():
    US, UA = payoff_table()
    assert US.tolist() == [[4, 9], [8, 3]]
    assert UA.tolist() == [[3, 9], [8, 4]]


def test_pure_nash():
    US, UA = payoff_table()
    assert pure_nash(US, UA, S_ACTIONS, A_ACTIONS) == [("D", "C"), ("Q", "F")]


def test_mixed_nash_matches_paper():
    alpha, beta = mixed_nash_2x2(*payoff_table())
    assert alpha == pytest.approx(2 / 5) and beta == pytest.approx(3 / 5)


@pytest.mark.parametrize("P", [4, 10, 25])
def test_mixed_formula_general_P(P):
    alpha, beta = mixed_nash_2x2(*payoff_table(P))
    assert alpha == pytest.approx((P - 2) / (2 * P)) and beta == pytest.approx((P + 2) / (2 * P))


def test_small_P_has_one_equilibrium():
    US, UA = payoff_table(P=1.5)
    assert pure_nash(US, UA, S_ACTIONS, A_ACTIONS) == [("D", "C")]


TABLE_VII = [[(5.6, 4.6), (8.0, 7.8), (5.0, 4.2), (7.4, 7.4)],
             [(7.2, 7.0), (3.2, 3.8), (8.2, 8.2), (4.2, 5.0)],
             [(4.8, 4.0), (8.8, 8.8), (3.8, 3.2), (7.8, 8.0)],
             [(6.4, 6.4), (4.0, 4.8), (7.0, 7.2), (4.6, 5.6)]]


def test_table_VII():
    US, UA = bayesian_table(0.2, 0.2)
    for r in range(4):
        for c in range(4):
            assert (US[r, c], UA[r, c]) == pytest.approx(TABLE_VII[r][c])


def test_bayesian_equilibria():
    assert bayesian_nash(0.2, 0.2) == [("DQ", "CF"), ("QD", "FC")]


def test_bayesian_equilibria_always_coordinate():
    for p, q in [(0.1, 0.3), (0.2, 0.2), (0.7, 0.4)]:
        for s, a in [("DQ", "CF"), ("QD", "FC")]:
            assert coherent_probability(s, a, p, q) == pytest.approx(1.0)


def test_rejected_items_moved_fast():
    for kind in ["s", "c"]:
        for m in ["D", "Q"]:
            assert arm_choice(m, kind, "fail", PARAMS) == "F"


def test_complex_items_get_detailed_scan():
    assert solve("c", dict(PARAMS, rho=0.10))["scan"] == "D"


def test_simple_items_get_quick_scan():
    assert solve("s", dict(PARAMS, rho=0.10))["scan"] == "Q"
