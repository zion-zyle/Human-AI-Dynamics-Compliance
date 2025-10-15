import matplotlib.pyplot as plt
import numpy as np
import os

def plot_simulation(simulator, save=False, filename=None):
    fig, axs = plt.subplots(4, 1, figsize=(12, 12))

    # 0) 제안 vs 실제 행동(GT)
    axs[0].plot(simulator.suggestion_trace, label='Agent Suggestion (numeric)')
    if simulator.ground_truth_action_trace:
        axs[0].scatter(range(len(simulator.ground_truth_action_trace)), simulator.ground_truth_action_trace,
                       label='GT User Action', s=12)
    if simulator.goal_trace:
        axs[0].plot(simulator.goal_trace, linestyle=':', label='Goal (dynamic)')
    axs[0].set_ylabel("Value (1~5)")
    axs[0].set_title("Suggestion vs GT Actions (with Goal)")
    axs[0].legend()
    axs[0].grid(True)

    # 1) Reward
    axs[1].plot(simulator.reward_trace, label='Reward')
    axs[1].set_ylabel("Reward")
    axs[1].set_title("Reward Dynamics")
    axs[1].legend()
    axs[1].grid(True)

    # 2) Compliance (GT 기반)
    axs[2].plot(simulator.compliance_trace_raw, label='Compliance (raw)')
    axs[2].plot(simulator.compliance_trace, label='Compliance (EMA)', linestyle='--')
    axs[2].plot(simulator.estimated_compliance_trace, label='Estimated Compliance (agent)', linestyle='-.')
    axs[2].set_ylabel("Compliance")
    axs[2].set_title("Compliance Over Time (GT-based)")
    axs[2].legend()
    axs[2].grid(True)

    # 3) 목표 변화
    if simulator.goal_trace:
        axs[3].plot(simulator.goal_trace, label='Goal')
    axs[3].set_ylabel("Goal")
    axs[3].set_title("Goal Over Time")
    axs[3].legend()
    axs[3].grid(True)

    plt.xlabel("Session")
    plt.tight_layout()

    if save and filename:
        png_dir = "plots"
        os.makedirs(png_dir, exist_ok=True)
        png_path = os.path.join(png_dir, f"{filename}.png")
        plt.savefig(png_path)
        return png_path

    plt.show()
