"""
defect_game.py
OUR EXTENSION: a defect-aware inspection game.

What the paper misses: its payoffs never depend on whether the item is
actually defective, the scan mode does not change accuracy, and robots move
at the same time. Here:

  1. Nature decides if the item is defective (probability rho).
     Whether it is simple ('s') or complex ('c') is visible to both robots.
  2. Scanner S picks D or Q. A detailed scan catches more defects.
  3. The scan gives a verdict: 'pass' or 'fail'.
  4. Arm A SEES the verdict, then picks F or C.

Because A moves after seeing the verdict, we solve it backwards
(backward induction): first find A's best move for each verdict,
then S picks the scan that is best for it, knowing how A will react.
"""

# ---------- parameters (our assumptions, NOT measured data) ----------
PARAMS = {
    "rho": 0.10,                                   # Pr[item is defective]
    # Pr[scan says 'fail' | item defective]  (true positive rate)
    "tpr": {("D", "s"): 0.98, ("D", "c"): 0.95, ("Q", "s"): 0.90, ("Q", "c"): 0.55},
    # Pr[scan says 'fail' | item good]       (false positive rate)
    "fpr": {("D", "s"): 0.02, ("D", "c"): 0.03, ("Q", "s"): 0.05, ("Q", "c"): 0.10},
    # Pr[item breaks while being moved]
    "brk": {("F", "s"): 0.02, ("F", "c"): 0.25, ("C", "s"): 0.01, ("C", "c"): 0.03},
    "L_miss": 30,     # cost of shipping a defective item
    "L_fa": 5,        # cost of rejecting a good item (false alarm)
    "V": 20,          # cost of breaking a good item on the way to shipping
    "salvage": 0.1,   # breaking a good item on the REJECT path costs only 10% of V
    "cost": {"D": 2, "Q": 1, "F": 1, "C": 2},     # same action costs as the paper
    "share": 0.5,     # how much each robot cares about the OTHER robot's job
}


def probabilities(m, kind, P):
    """Joint probabilities of (item status, verdict) for scan mode m."""
    rho, tpr, fpr = P["rho"], P["tpr"][(m, kind)], P["fpr"][(m, kind)]
    return {
        ("bad", "fail"): rho * tpr,
        ("bad", "pass"): rho * (1 - tpr),          # missed defect!
        ("good", "fail"): (1 - rho) * fpr,         # false alarm
        ("good", "pass"): (1 - rho) * (1 - fpr),
    }


def prob_good(m, kind, verdict, P):
    """Bayes' rule: Pr[item is good | verdict]."""
    pr = probabilities(m, kind, P)
    return pr[("good", verdict)] / (pr[("good", verdict)] + pr[("bad", verdict)])


def arm_choice(m, kind, verdict, P):
    """Arm's best move after seeing the verdict.
    Arm's cost = expected breakage loss + its own action cost."""
    weight = 1.0 if verdict == "pass" else P["salvage"]
    mu = prob_good(m, kind, verdict, P)
    expected_cost = {a: P["brk"][(a, kind)] * mu * P["V"] * weight + P["cost"][a] for a in ["F", "C"]}
    return min(expected_cost, key=expected_cost.get)


def evaluate(kind, m, arm_if_pass, arm_if_fail, P):
    """Expected losses when S scans with m and A follows the given moves."""
    pr = probabilities(m, kind, P)
    classification = P["L_miss"] * pr[("bad", "pass")] + P["L_fa"] * pr[("good", "fail")]
    breakage = (pr[("good", "pass")] * P["brk"][(arm_if_pass, kind)] * P["V"]
                + pr[("good", "fail")] * P["brk"][(arm_if_fail, kind)] * P["V"] * P["salvage"])
    p_pass = pr[("good", "pass")] + pr[("bad", "pass")]
    arm_cost = p_pass * P["cost"][arm_if_pass] + (1 - p_pass) * P["cost"][arm_if_fail]
    return {
        "scanner_cost": classification + P["share"] * breakage + P["cost"][m],   # what S minimises
        "team_loss": classification + breakage + P["cost"][m] + arm_cost,         # total for the factory
    }


def solve(kind, P=PARAMS):
    """Backward induction. Returns S's scan mode, A's move for each verdict, team loss."""
    best = None
    for m in ["D", "Q"]:
        a_pass = arm_choice(m, kind, "pass", P)
        a_fail = arm_choice(m, kind, "fail", P)
        res = evaluate(kind, m, a_pass, a_fail, P)
        if best is None or res["scanner_cost"] < best["scanner_cost"]:
            best = {"scan": m, "arm_if_pass": a_pass, "arm_if_fail": a_fail, **res}
    return best


def paper_fixed_policies(kind, P=PARAMS):
    """The paper's two coherent pairs used for EVERY item, ignoring the verdict."""
    return {"(D,C)": evaluate(kind, "D", "C", "C", P)["team_loss"],
            "(Q,F)": evaluate(kind, "Q", "F", "F", P)["team_loss"]}


if __name__ == "__main__":
    for rho in [0.02, 0.10, 0.30, 0.45]:
        P = dict(PARAMS, rho=rho)
        for kind, name in [("s", "simple "), ("c", "complex")]:
            r = solve(kind, P)
            fixed = paper_fixed_policies(kind, P)
            print(f"rho={rho:.2f} {name}: S scans {r['scan']}, A: pass->{r['arm_if_pass']} fail->{r['arm_if_fail']}"
                  f" | loss {r['team_loss']:.2f}  vs paper (D,C) {fixed['(D,C)']:.2f}, (Q,F) {fixed['(Q,F)']:.2f}")
