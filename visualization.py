from typing import List, Optional
import math

import matplotlib.pyplot as plt


def _ema(xs: List[float], alpha: float = 0.3) -> List[float]:
    if not xs:
        return []
    out = [xs[0]]
    for i in range(1, len(xs)):
        out.append(alpha * xs[i] + (1 - alpha) * out[-1])
    return out


def plot_simulation(
    suggestions: List[float],
    actions: List[float],
    rewards: List[float],
    compliances: List[float],
    est_compliances: Optional[List[float]] = None,
    title_suffix: str = "user",
    out_path: str = "simulation_plot.png",
) -> None:
    n = len(suggestions)
    xs = list(range(n))

    reward_ema = _ema(rewards, alpha=0.3)
    comp_ema = _ema(compliances, alpha=0.3)
    est_comp = est_compliances if est_compliances is not None else [float("nan")] * n

    fig = plt.figure(figsize=(14, 11))

    # 1) suggestion vs action
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(xs, suggestions, label="Agent Suggestion (numeric)")
    ax1.scatter(xs, actions, s=18, label="GT User Action")
    ax1.set_title(f"Suggestion vs Action — {title_suffix}")
    ax1.set_ylabel("Value (1~5)")
    ax1.set_ylim(1.0, 5.0)
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="upper right")

    # 2) reward
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(xs, rewards, alpha=0.35, label="Reward (raw)")
    ax2.plot(xs, reward_ema, label="Reward (EMA, α=0.3)")
    ax2.set_title("Reward Dynamics (smoothed)")
    ax2.set_ylabel("Reward")
    ax2.grid(True, alpha=0.35)
    ax2.legend(loc="upper right")

    # 3) compliance
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(xs, compliances, alpha=0.35, label="Compliance (raw)")
    ax3.plot(xs, comp_ema, label="Compliance (EMA, smoothed)")
    ax3.plot(xs, est_comp, linestyle="--", label="Estimated Compliance (agent)")
    ax3.set_title("Compliance Dynamics")
    ax3.set_xlabel("Session")
    ax3.set_ylabel("Compliance")
    ax3.set_ylim(0.0, 1.0)
    ax3.grid(True, alpha=0.35)
    ax3.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
