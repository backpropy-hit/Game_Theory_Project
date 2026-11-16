"""
make_figures.py
Generates every figure in figures/ and the result tables in results/.
Run:  python make_figures.py
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_game import (payoff_table, mixed_nash_2x2, bayesian_table, bayesian_nash,
                        S_ACTIONS, A_ACTIONS, S_PLANS, A_PLANS)
from learning import fictitious_play, run_many
from defect_game import PARAMS, solve, paper_fixed_policies

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}.png", dpi=200)
    fig.savefig(f"figures/{name}.pdf")
    plt.close(fig)
    print("saved", name)


def heatmap(ax, values, text, rows, cols, title, boxes=()):
    ax.imshow(values, cmap="Blues")
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, text[i][j], ha="center", va="center",
                    color="white" if values[i][j] > values.mean() else "black",
                    fontweight="bold" if (i, j) in boxes else "normal")
    for i, j in boxes:
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=2.5, ec="red"))
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title(title)


# ---------- Fig 1: the standard game ----------
US, UA = payoff_table()
fig, ax = plt.subplots(figsize=(4, 3.2))
text = [[f"{US[i, j]:.0f}, {UA[i, j]:.0f}" for j in range(2)] for i in range(2)]
heatmap(ax, US + UA, text, ["D (detailed)", "Q (quick)"], ["F (fast)", "C (cautious)"],
        "Standard game (P = 10): $u_S$, $u_A$\nred boxes = pure Nash equilibria", boxes=[(0, 1), (1, 0)])
ax.set_xlabel("Arm A")
ax.set_ylabel("Scanner S")
save(fig, "fig1_standard_game")

# ---------- Fig 2: best responses ----------
alpha, beta = mixed_nash_2x2(US, UA)
fig, ax = plt.subplots(figsize=(4, 3.4))
ax.plot([0, beta, beta, 1], [1, 1, 0, 0], lw=2, label="S's best response")
ax.plot([1, 1, 0, 0], [0, alpha, alpha, 1], lw=2, ls="--", label="A's best response")
for x, y, label in [(0, 1, "(D,C)  9,9"), (1, 0, "(Q,F)  8,8"), (beta, alpha, "mixed  6,6")]:
    ax.plot(x, y, "ko", ms=7)
    ax.annotate(label, (x, y), (x + 0.04 if x < 0.9 else x - 0.3, y + 0.05 if y < 0.9 else y - 0.1))
ax.set_xlabel("Pr[A plays F]")
ax.set_ylabel("Pr[S plays D]")
ax.set_title("Best responses: 3 Nash equilibria")
ax.legend(loc="lower left", fontsize=8)
ax.grid(alpha=0.3)
save(fig, "fig2_best_response")

# ---------- Fig 3: Table VII ----------
BS, BA = bayesian_table(0.2, 0.2)
bne = [(S_PLANS.index(s), A_PLANS.index(a)) for s, a in bayesian_nash(0.2, 0.2)]
fig, ax = plt.subplots(figsize=(5, 3.8))
text = [[f"{BS[r, c]:.1f}, {BA[r, c]:.1f}" for c in range(4)] for r in range(4)]
heatmap(ax, BS + BA, text, S_PLANS, A_PLANS, "Bayesian game, p = q = 0.2 (paper's Table VII)\nred boxes = Bayesian Nash equilibria", boxes=bne)
ax.set_xlabel("Arm plan (action if Mistaken, action if Right)")
ax.set_ylabel("Scanner plan")
save(fig, "fig3_bayesian_table")

# ---------- Fig 4: Bayesian equilibria for all p and q ----------
grid = np.linspace(0.01, 0.99, 50)
extra = np.zeros((50, 50))
rows = []
for a, q in enumerate(grid):
    for b, p in enumerate(grid):
        eqs = bayesian_nash(p, q)
        extra[a, b] = 1 if ("DD", "CC") in eqs else 0
        rows.append([round(p, 3), round(q, 3), " ".join(f"({s},{t})" for s, t in eqs)])
with open("results/bayesian_equilibria_pq.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["p", "q", "equilibria"])
    w.writerows(rows)
fig, ax = plt.subplots(figsize=(4, 3.6))
ax.imshow(extra, origin="lower", extent=[0, 1, 0, 1], cmap="Oranges", vmin=0, vmax=1.5)
ax.plot(0.2, 0.2, "k*", ms=10, label="paper's case")
ax.text(0.5, 0.85, "(DQ,CF) and (QD,FC)\neverywhere", ha="center")
ax.text(0.5, 0.5, "+ (DD,CC)", ha="center", fontsize=8)
ax.set_xlabel("p = Pr[scanner mistaken]")
ax.set_ylabel("q = Pr[arm mistaken]")
ax.set_title("Bayesian Nash equilibria for all p, q")
ax.legend(loc="lower right", fontsize=8)
save(fig, "fig4_bayesian_pq_map")

# ---------- Fig 5: fictitious play ----------
share_DC, share_QF = run_many(n_runs=100)
fig, ax = plt.subplots(figsize=(4, 3.6))
rng = np.random.default_rng(1)
for _ in range(30):
    beliefs, final = fictitious_play(US, UA, 150, seed=rng.integers(1_000_000))
    color = "green" if final == (0, 1) else "red"
    ax.plot(beliefs[:, 1], beliefs[:, 0], color=color, lw=0.8, alpha=0.7)
    ax.plot(beliefs[0, 1], beliefs[0, 0], "o", color=color, ms=3)
ax.plot(beta, alpha, "kx", ms=8, label="mixed NE")
ax.set_xlabel("belief: Pr[A plays F]")
ax.set_ylabel("belief: Pr[S plays D]")
ax.set_title(f"Fictitious play (100 runs): {share_DC:.0%} reach (D,C)\ngreen = ends at (D,C), red = ends at (Q,F)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
save(fig, "fig5_fictitious_play")

# ---------- Fig 6: extension - which scan does S choose? ----------
rhos = np.linspace(0.005, 0.5, 60)
tprs = np.linspace(0.3, 0.98, 60)
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=True)
for ax, kind, name in [(axes[0], "s", "simple item"), (axes[1], "c", "complex item")]:
    choice = np.zeros((60, 60))
    for i, t in enumerate(tprs):
        for j, rho in enumerate(rhos):
            P = dict(PARAMS, rho=rho, tpr=dict(PARAMS["tpr"]))
            P["tpr"][("Q", kind)] = t
            choice[i, j] = 1 if solve(kind, P)["scan"] == "D" else 0
    ax.imshow(choice, origin="lower", extent=[rhos[0], rhos[-1], tprs[0], tprs[-1]],
              aspect="auto", cmap="coolwarm", vmin=-0.5, vmax=1.5)
    ax.axhline(PARAMS["tpr"][("Q", kind)], color="black", ls="--", lw=1)
    ax.text(0.3, 0.4, "detailed scan (D)", ha="center")
    ax.text(0.1, 0.93, "quick scan (Q)", ha="center")
    ax.set_title(f"Scanner's choice, {name}")
    ax.set_xlabel("defect probability rho")
axes[0].set_ylabel("quick-scan detection rate")
save(fig, "fig6_extension_scan_choice")

# ---------- Fig 7: extension vs the paper ----------
rows = []
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
for ax, kind, name in [(axes[0], "s", "simple item"), (axes[1], "c", "complex item")]:
    ours, dc, qf = [], [], []
    for rho in rhos:
        P = dict(PARAMS, rho=rho)
        r = solve(kind, P)
        fixed = paper_fixed_policies(kind, P)
        ours.append(r["team_loss"]); dc.append(fixed["(D,C)"]); qf.append(fixed["(Q,F)"])
        rows.append([kind, round(rho, 4), r["scan"], r["arm_if_pass"], r["arm_if_fail"],
                     round(r["team_loss"], 4), round(fixed["(D,C)"], 4), round(fixed["(Q,F)"], 4)])
    ax.plot(rhos, qf, "--", label="paper: always (Q,F)")
    ax.plot(rhos, dc, "-.", label="paper: always (D,C)")
    ax.plot(rhos, ours, lw=2.5, label="our equilibrium")
    ax.set_title(name)
    ax.set_xlabel("defect probability rho")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("expected loss per item")
axes[1].legend(fontsize=8)
save(fig, "fig7_extension_vs_paper")
with open("results/extension_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["item", "rho", "scan", "arm_if_pass", "arm_if_fail", "our_loss", "paper_DC_loss", "paper_QF_loss"])
    w.writerows(rows)

print(f"\nFictitious play: {share_DC:.0%} to (D,C), {share_QF:.0%} to (Q,F)")
