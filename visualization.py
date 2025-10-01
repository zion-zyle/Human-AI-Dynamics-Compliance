# visualization.py

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_simulation(simulator, save=False, filename=None):
    fig, axs = plt.subplots(5, 1, figsize=(12, 14))

    # 0) 제안 vs 행동(추론/실제)
    axs[0].plot(simulator.suggestion_trace, label='Agent Suggestion (numeric)')
    if simulator.inferred_action_trace:
        axs[0].plot(simulator.inferred_action_trace, label='Inferred User Action', linestyle='--')
    if simulator.ground_truth_action_trace:
        axs[0].scatter(range(len(simulator.ground_truth_action_trace)), simulator.ground_truth_action_trace,
                       label='GT User Action', s=12)
    axs[0].axhline(simulator.agent.goal_behavior, linestyle=':', color='red', label='Agent Goal G')
    axs[0].set_ylabel("Value (1~5)")
    axs[0].set_title("Suggestion vs User Actions")
    axs[0].legend()
    axs[0].grid(True)

    # 1) Reward
    axs[1].plot(simulator.reward_trace, label='Reward')
    axs[1].set_ylabel("Reward")
    axs[1].set_title("Reward Dynamics")
    axs[1].legend()
    axs[1].grid(True)

    # 2) Compliance
    axs[2].plot(simulator.compliance_trace, label='Compliance (est)')
    axs[2].plot(simulator.estimated_compliance_trace, label='Estimated Compliance (agent)', linestyle='--')
    axs[2].set_ylabel("Compliance")
    axs[2].set_title("Compliance Over Time")
    axs[2].legend()
    axs[2].grid(True)

    # 3) Mu estimate
    axs[3].plot(simulator.estimated_mu_trace, label=r'Estimated $\hat{\mu}$')
    axs[3].set_ylabel("Mean")
    axs[3].set_title(r"Estimated Behavior Mean ($\hat{\mu}$)")
    axs[3].legend()
    axs[3].grid(True)

    # 4) Max Q
    if simulator.q_value_trace:
        q_values_over_time = np.array(simulator.q_value_trace)
        max_q_values = np.max(q_values_over_time, axis=1)
        axs[4].plot(max_q_values, label='Max Q-value')
    axs[4].set_ylabel("Max Q")
    axs[4].set_title("Q-value Over Time")
    axs[4].legend()
    axs[4].grid(True)

    plt.xlabel("Session")
    plt.tight_layout()

    if save and filename:
        png_dir = "plots"
        os.makedirs(png_dir, exist_ok=True)
        png_path = os.path.join(png_dir, f"{filename}.png")
        plt.savefig(png_path)
        return png_path

    plt.show()
