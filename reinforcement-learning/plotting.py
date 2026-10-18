"""Presentation figures; kept separate from the learning algorithm."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "q-learning-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from q_learning import MAP, START, GOAL, ACTIONS, ACTION_NAMES


def draw_grid(ax):
    for row in range(5):
        for col in range(5):
            state = row * 5 + col
            color = "#283444" if MAP[row][col] == "#" else "#f3f6fa"
            if state == START:
                color = "#c9e6ff"
            if state == GOAL:
                color = "#bcf0d4"
            ax.add_patch(Rectangle((col-.5, row-.5), 1, 1, facecolor=color, edgecolor="white", lw=2))
    ax.set(xlim=(-.5, 4.5), ylim=(4.5, -.5), xticks=range(5), yticks=range(5), xlabel="Column", ylabel="Row")
    ax.set_aspect("equal")
    ax.text(0, -.40, "START", ha="center", va="center", fontsize=7, weight="bold")
    ax.text(4, 4.3, "GOAL", ha="center", fontsize=8, weight="bold")


def save_figures(q, history, before, after, output):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for ax, result, title in zip(axes, (before, after), ("Before learning: all Q = 0", "After learning: greedy policy")):
        draw_grid(ax)
        path, total, success = result
        rows, cols = zip(*(divmod(s, 5) for s in path))
        ax.plot(cols, rows, "o-", color="#2476b8", ms=4, alpha=.7)
        ax.set_title(f"{title}\n{len(path)-1} steps | return {total} | {'goal reached' if success else 'step cap reached'}")
    fig.suptitle("Same maze, same rewards: learning changes the actions", weight="bold")
    fig.savefig(output / "routes.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    draw_grid(axes[0])
    draw_grid(axes[1])
    for state in range(25):
        row, col = divmod(state, 5)
        if MAP[row][col] == "#" or state == GOAL:
            continue
        best = np.flatnonzero(np.isclose(q[state], q[state].max(), atol=1e-6, rtol=0))
        for action in best:
            dr, dc = ACTIONS[action]
            axes[0].arrow(col, row, dc*.27, dr*.27, width=.018, head_width=.14, color="#2476b8", length_includes_head=True)
        for action, (dr, dc) in enumerate(ACTIONS):
            axes[1].text(col+dc*.29, row+dr*.27, f"{q[state, action]:.1f}", ha="center", va="center", fontsize=8,
                         color="#067344" if action in best else "#586577")
    axes[0].set_title("Greedy actions (ties shown)")
    axes[1].set_title("Q table: up / right / down / left")
    fig.suptitle("Learned action values on this fixed maze", weight="bold")
    fig.savefig(output / "policy_and_q_values.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    episodes = np.array([h["episode"] for h in history])
    returns = np.array([h["train_return"] for h in history])
    window = min(50, len(history))
    axes[0].plot(episodes, returns, alpha=.18, color="#2476b8", label="Training episodes")
    axes[0].plot(episodes[window-1:], np.convolve(returns, np.ones(window)/window, mode="valid"), color="#2476b8", label=f"{window}-episode mean")
    axes[0].plot(episodes, [h["eval_return"] for h in history], color="#159568", label="Greedy evaluation", alpha=.8)
    axes[0].set(title="Return: exploration vs. evaluation", ylabel="Undiscounted reward sum")
    axes[1].plot(episodes, [h["epsilon"] for h in history], color="#bd7028")
    axes[1].set(title="Exploration probability", ylabel="Epsilon", ylim=(0, 1.05))
    for name in ACTION_NAMES:
        axes[2].plot(episodes, [h[f"start_q_{name}"] for h in history], label=name)
    axes[2].set(title="Four Q values at the start", ylabel="Estimated discounted return")
    for ax in axes:
        ax.set_xlabel("Training episode")
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.savefig(output / "learning_curves.png", dpi=180)
    plt.close(fig)
