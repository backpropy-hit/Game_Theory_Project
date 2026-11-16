"""
paper_game.py
Re-implementation of the game from the paper:
  Gindullina et al., "A Game Theory Model for Multi Robot Cooperation
  in Industry 4.0 Scenarios", IEEE APCCAS 2021.

Players : S = scanner robot  -> actions D (detailed scan), Q (quick scan)
          A = robotic arm    -> actions F (fast),          C (cautious)
Payoff  : u = P*r - cost + bonus      (Eq. 1 in the paper)
          r = 1   if the pair is "coherent" ((D,C) or (Q,F))
          r = 0.5 otherwise
"""
import numpy as np

# ---- values from Table I of the paper ----
COST = {"D": 2, "Q": 1, "F": 1, "C": 2}
BONUS = {"D": 1, "Q": -1, "F": -1, "C": 1}
R_COHERENT = 1.0
R_NOT = 0.5

S_ACTIONS = ["D", "Q"]   # rows
A_ACTIONS = ["F", "C"]   # columns

NORMAL_PAIRS = [("D", "C"), ("Q", "F")]    # coherent pairs (Table III / IV)
FLIPPED_PAIRS = [("D", "F"), ("Q", "C")]   # coherent pairs when types differ (Table V)


def payoff_table(P=10, coherent_pairs=NORMAL_PAIRS):
    """Return two 2x2 arrays: US[i][j] = scanner payoff, UA[i][j] = arm payoff.
    Row i = scanner action, column j = arm action."""
    US = np.zeros((2, 2))
    UA = np.zeros((2, 2))
    for i, s in enumerate(S_ACTIONS):
        for j, a in enumerate(A_ACTIONS):
            r = R_COHERENT if (s, a) in coherent_pairs else R_NOT
            US[i, j] = P * r - COST[s] + BONUS[s]
            UA[i, j] = P * r - COST[a] + BONUS[a]
    return US, UA


def pure_nash(US, UA, row_names, col_names):
    """Check every cell: it is a Nash equilibrium if the row player can't do
    better in that column AND the column player can't do better in that row."""
    eq = []
    for i in range(US.shape[0]):
        for j in range(US.shape[1]):
            row_best = US[i, j] >= US[:, j].max() - 1e-9
            col_best = UA[i, j] >= UA[i, :].max() - 1e-9
            if row_best and col_best:
                eq.append((row_names[i], col_names[j]))
    return eq


def mixed_nash_2x2(US, UA):
    """Indifference method for a 2x2 game.
    alpha = Pr[S plays first row (D)], beta = Pr[A plays first column (F)].
    beta makes S indifferent between D and Q; alpha makes A indifferent between F and C."""
    beta = (US[1, 1] - US[0, 1]) / (US[0, 0] - US[0, 1] - US[1, 0] + US[1, 1])
    alpha = (UA[1, 1] - UA[1, 0]) / (UA[0, 0] - UA[1, 0] - UA[0, 1] + UA[1, 1])
    return alpha, beta


def risk_dominance_score(US, UA, i, j):
    """Harsanyi-Selten: product of what each player loses by deviating alone
    from the equilibrium (i, j). Bigger = 'safer' equilibrium."""
    loss_S = US[i, j] - US[1 - i, j]
    loss_A = UA[i, j] - UA[i, 1 - j]
    return loss_S * loss_A


# ------------------------------------------------------------------
# Bayesian game (Section III of the paper)
# Each robot is Mistaken (M) with prob p (scanner) / q (arm), else Right (R).
# A strategy is a plan "XY": X = action if Mistaken, Y = action if Right.
# (We found this order by matching the paper's Table VII.)
# ------------------------------------------------------------------
S_PLANS = ["DD", "DQ", "QD", "QQ"]
A_PLANS = ["FF", "FC", "CF", "CC"]
TYPES = ["M", "R"]


def bayesian_table(p, q, P=10):
    """Expected (ex-ante) payoffs for every pair of plans -> two 4x4 arrays."""
    same_S, same_A = payoff_table(P, NORMAL_PAIRS)    # types MM or RR
    diff_S, diff_A = payoff_table(P, FLIPPED_PAIRS)   # types MR or RM
    prob = {"M": p, "R": 1 - p}
    probA = {"M": q, "R": 1 - q}

    US = np.zeros((4, 4))
    UA = np.zeros((4, 4))
    for r, planS in enumerate(S_PLANS):
        for c, planA in enumerate(A_PLANS):
            for k, tS in enumerate(TYPES):        # k = 0 for M, 1 for R
                for m, tA in enumerate(TYPES):
                    pr = prob[tS] * probA[tA]
                    i = S_ACTIONS.index(planS[k])  # action this type plays
                    j = A_ACTIONS.index(planA[m])
                    if tS == tA:
                        US[r, c] += pr * same_S[i, j]
                        UA[r, c] += pr * same_A[i, j]
                    else:
                        US[r, c] += pr * diff_S[i, j]
                        UA[r, c] += pr * diff_A[i, j]
    return US, UA


def bayesian_nash(p, q, P=10):
    US, UA = bayesian_table(p, q, P)
    return pure_nash(US, UA, S_PLANS, A_PLANS)


def coherent_probability(planS, planA, p, q):
    """Probability that the robots actually get the coherent reward r_c."""
    total = 0.0
    prob = {"M": p, "R": 1 - p}
    probA = {"M": q, "R": 1 - q}
    for k, tS in enumerate(TYPES):
        for m, tA in enumerate(TYPES):
            pair = (planS[k], planA[m])
            pairs = NORMAL_PAIRS if tS == tA else FLIPPED_PAIRS
            if pair in pairs:
                total += prob[tS] * probA[tA]
    return total


if __name__ == "__main__":
    US, UA = payoff_table()
    print("Standard game (Table III)  [S payoff, A payoff]")
    for i, s in enumerate(S_ACTIONS):
        print(f"  {s}: " + "   ".join(f"{A_ACTIONS[j]}=({US[i,j]:.0f},{UA[i,j]:.0f})" for j in range(2)))
    print("Pure NE:", pure_nash(US, UA, S_ACTIONS, A_ACTIONS))
    a, b = mixed_nash_2x2(US, UA)
    print(f"Mixed NE: Pr[D] = {a:.2f}, Pr[F] = {b:.2f}")
    print(f"Mixed NE payoffs: S = {b*US[0,0]+(1-b)*US[0,1]:.1f}, A = {a*UA[0,0]+(1-a)*UA[1,0]:.1f}")
    print("Risk-dominance score (D,C) =", risk_dominance_score(US, UA, 0, 1),
          " (Q,F) =", risk_dominance_score(US, UA, 1, 0))

    print("\nBayesian game, p = q = 0.2 (Table VII)")
    BS, BA = bayesian_table(0.2, 0.2)
    print("      " + "        ".join(A_PLANS))
    for r, plan in enumerate(S_PLANS):
        print(f"  {plan}  " + "  ".join(f"{BS[r,c]:.1f},{BA[r,c]:.1f}" for c in range(4)))
    for planS, planA in bayesian_nash(0.2, 0.2):
        print(f"BNE ({planS},{planA}): Pr[coherent reward] = {coherent_probability(planS, planA, 0.2, 0.2):.2f}")
