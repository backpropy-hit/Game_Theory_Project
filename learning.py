"""
learning.py
Repeated play with FICTITIOUS PLAY (Brown, 1951):
every round, each robot looks at how often the other robot used each action
in the past, and plays its best response to those frequencies.

Question: if the robots play the paper's game many times without talking,
do they settle on an equilibrium, and which one?
"""
import numpy as np
from paper_game import payoff_table


def fictitious_play(US, UA, rounds=200, seed=None):
    rng = np.random.default_rng(seed)
    # pretend history: random starting counts so each run starts with different beliefs
    count_S = rng.random(2) * 3 + 0.1   # how often S played D, Q
    count_A = rng.random(2) * 3 + 0.1   # how often A played F, C
    beliefs = []
    for t in range(rounds):
        belief_A = count_A / count_A.sum()   # S's belief about A
        belief_S = count_S / count_S.sum()   # A's belief about S
        beliefs.append((belief_S[0], belief_A[0]))   # Pr[D], Pr[F]

        s_values = US @ belief_A             # expected payoff of D and Q
        a_values = belief_S @ UA             # expected payoff of F and C
        i = int(np.argmax(s_values))
        j = int(np.argmax(a_values))
        count_S[i] += 1
        count_A[j] += 1
    return np.array(beliefs), (i, j)


def run_many(n_runs=100, rounds=200, seed=0):
    US, UA = payoff_table()
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_runs):
        _, final = fictitious_play(US, UA, rounds, seed=rng.integers(1_000_000))
        results.append(final)
    share_DC = np.mean([r == (0, 1) for r in results])
    share_QF = np.mean([r == (1, 0) for r in results])
    return share_DC, share_QF


if __name__ == "__main__":
    dc, qf = run_many()
    print(f"Out of 100 runs: {dc:.0%} end at (D,C), {qf:.0%} end at (Q,F)")
