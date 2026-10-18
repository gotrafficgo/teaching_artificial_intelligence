"""A small, reproducible tabular Q-learning lesson. Run this file directly."""

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import numpy as np


# State = row * 5 + column. Action order: up, right, down, left.
MAP = ("S....", ".###.", "...#.", ".#...", "...#G")
ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_NAMES = ("up", "right", "down", "left")
START, GOAL = 0, 24
N_STATES = 25


def step(state, action):
    """Environment: walls leave the robot in place; only the goal terminates."""
    if state == GOAL:
        return GOAL, 0, True
    row, col = divmod(state, 5)
    dr, dc = ACTIONS[action]
    nr, nc = row + dr, col + dc
    if 0 <= nr < 5 and 0 <= nc < 5 and MAP[nr][nc] != "#":
        state = nr * 5 + nc
    terminated = state == GOAL
    reward = 10 if terminated else -1
    return state, reward, terminated


def choose_action(q, state, epsilon, rng):
    """Explore randomly, or exploit; break ties randomly to avoid direction bias."""
    if rng.random() < epsilon:
        return int(rng.integers(4))
    best = np.flatnonzero(q[state] == q[state].max())
    return int(rng.choice(best))


def update_q(q, state, action, reward, next_state, terminated, alpha, gamma):
    """One transition changes exactly one table entry."""
    target = reward if terminated else reward + gamma * q[next_state].max()
    q[state, action] += alpha * (target - q[state, action])


def rollout(q, seed=123, max_steps=100):
    """Evaluate without exploration or updates. Tied best actions remain random."""
    rng = np.random.default_rng(seed)
    state, total = START, 0
    path = [state]
    for _ in range(max_steps):
        action = choose_action(q, state, 0, rng)
        state, reward, terminated = step(state, action)
        path.append(state)
        total += reward
        if terminated:
            break
    return path, total, state == GOAL


def train(episodes=1500, seed=7, max_steps=100, alpha=0.2, gamma=0.95):
    rng = np.random.default_rng(seed)
    q = np.zeros((N_STATES, 4))
    history, snapshots = [], {0: q.copy()}
    for episode in range(1, episodes + 1):
        epsilon = max(0.05, 1.0 * 0.995 ** (episode - 1))
        state, total = START, 0
        for steps in range(1, max_steps + 1):
            action = choose_action(q, state, epsilon, rng)
            next_state, reward, terminated = step(state, action)
            update_q(q, state, action, reward, next_state, terminated, alpha, gamma)
            state = next_state
            total += reward
            if terminated:
                break
        # The step cap truncates data collection; it is NOT a terminal state.
        # Therefore its final update still includes the next state's value.
        eval_path, eval_return, eval_success = rollout(q, max_steps=max_steps)
        history.append({"episode": episode, "epsilon": epsilon,
                        "train_return": total, "train_steps": steps,
                        "train_success": int(terminated),
                        "eval_return": eval_return, "eval_steps": len(eval_path) - 1,
                        "eval_success": int(eval_success),
                        **{f"start_q_{name}": q[START, a]
                           for a, name in enumerate(ACTION_NAMES)}})
        if episode in (1, 10, 50, 200, episodes):
            snapshots[episode] = q.copy()
        if episode == 1 or episode % 300 == 0 or episode == episodes:
            print(f"Episode {episode:4d} | epsilon={epsilon:.3f} | "
                  f"training return={total:4d} | greedy evaluation: "
                  f"{'goal' if eval_success else 'step cap'}, {len(eval_path)-1} steps")
    return q, history, snapshots


def shortest_path_length():
    """BFS reference for verification only; the learner never uses this result."""
    queue, seen = deque([(START, 0)]), {START}
    while queue:
        state, distance = queue.popleft()
        if state == GOAL:
            return distance
        for action in range(4):
            next_state, _, _ = step(state, action)
            if next_state not in seen:
                seen.add(next_state)
                queue.append((next_state, distance + 1))
    raise ValueError("Goal is unreachable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    # A transparent hand-calculation example; this table is separate from training.
    demo = np.zeros((N_STATES, 4))
    update_q(demo, 19, 2, 10, GOAL, True, 0.2, 0.95)
    update_q(demo, 14, 2, -1, 19, False, 0.2, 0.95)
    print("Hand calculation (separate from training):")
    print(f"  Next to goal: Q(19, down) = 0 + 0.2*(10-0) = {demo[19, 2]:.2f}")
    print(f"  One step earlier: Q(14, down) = 0 + 0.2*(-1+0.95*2-0) = {demo[14, 2]:.2f}")

    before = rollout(np.zeros((N_STATES, 4)))
    q, history, snapshots = train(args.episodes, args.seed)
    after = rollout(q)
    reference = shortest_path_length()
    summary = {"seed": args.seed, "episodes": args.episodes,
               "before": {"steps": len(before[0])-1, "return": before[1], "success": before[2]},
               "after": {"steps": len(after[0])-1, "return": after[1], "success": after[2],
                         "path_row_col": [list(divmod(s, 5)) for s in after[0]]},
               "shortest_path_steps": reference,
               "learned_shortest_path": bool(after[2] and len(after[0])-1 == reference)}
    np.save(args.output / "q_table.npy", q)
    np.savez(args.output / "q_snapshots.npz", **{f"episode_{n}": v for n, v in snapshots.items()})
    with (args.output / "training.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    from plotting import save_figures
    save_figures(q, history, before, after, args.output)
    print(json.dumps(summary, indent=2))
    print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
