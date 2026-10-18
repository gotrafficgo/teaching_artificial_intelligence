# Q-learning: Learning to Navigate a Maze

A tabular Q-learning example for senior undergraduate students. No dataset, neural network, or online service is required.

## Run the example

Run from the repository root (the shared environment already includes the dependencies):

```bash
python3 reinforcement-learning/q_learning.py
```

If dependencies are missing:

```bash
python3 -m pip install -r reinforcement-learning/requirements.txt
```

By default, the example trains for 1,500 episodes with a random seed of 7; results are saved to this directory's `results/`.
It prints a worked calculation and training progress, then saves plots without opening a graphics window.

```bash
python3 reinforcement-learning/q_learning.py --episodes 3000 --seed 42
```

## Learning from an Example

The robot starts at S, navigates around the obstacles (#), and reaches G:

```text
S . . . .
. # # # .
. . . # .
. # . . .
. . . # G
```

- **State**: The current position, `state = row * 5 + column`, with rows and columns starting at 0.
- **Action**: Up, right, down, and left, numbered 0, 1, 2, and 3, respectively.
- **Reward**: +10 for reaching the goal and −1 for every other step. Hitting a wall or obstacle leaves the robot in place and also gives −1.
- **Episode**: Starts at the initial position and ends upon reaching the goal; at most 100 steps are collected before the environment resets.
- **Q table**: 25 × 4. Obstacle and goal states are not learned or used for action selection.
- **Policy**: Selects actions based on the Q table; training uses ε-greedy exploration.

### Suggested demonstration order

1. Open `results/routes.png` to compare the routes before and after training.
   Before training, all Q values are 0, so ties are broken randomly and behavior is random.
   Routes may overlap, and the step count in the title includes repeated moves and collisions with walls.
2. Open `results/policy_and_q_values.png` to explain the scores for the four actions in each cell and the best action.
   The arrows come from the estimated Q table; one successful route does not mean that the values at every position have fully converged.
3. Use the two worked calculations printed in the terminal to explain how reward information propagates backward.
4. Open `results/learning_curves.png` to compare training with exploration and evaluation with exploration disabled.
   Evaluation does not modify the Q table or consume the random numbers used for training. Ties between optimal actions are broken randomly.

## Main Code in the Example

The main algorithm is in `q_learning.py`; plotting is separated into `plotting.py`.

| Function | Teaching focus |
| --- | --- |
| `step` | State transitions and rewards are environment rules |
| `choose_action` | ε-greedy exploration and random tie-breaking |
| `update_q` | Each experience updates a single Q value |
| `train` | Repeated interaction, updates, and resets |
| `rollout` | Uses the learned policy without further updates |

Core update:

```python
target = reward if terminated else reward + gamma * q[next_state].max()
q[state, action] += alpha * (target - q[state, action])
```

- `alpha = 0.2`: How much to move toward the new estimate.
- `gamma = 0.95`: The discount factor for future rewards.
- `epsilon`: Gradually decreases from 1.0, retaining a minimum exploration probability of 0.05.
- **The goal differs from the step limit**: Reaching the goal has no future value; 100 steps is only a sampling limit.
  Updates still include the value of the next state rather than incorrectly treating the limit as a terminal state.

The worked example uses a separate table and does not inject prior knowledge into actual training:

```text
Q(19, down) = 0 + 0.2 × (10 − 0) = 2.00
Q(14, down) = 0 + 0.2 × (−1 + 0.95 × 2 − 0) = 0.18
```

The undiscounted sum of rewards per episode is used for plotting; Q values estimate discounted cumulative rewards, so the two are different.
The total reward for reaching the goal in 8 steps is `7 × (−1) + 10 = 3`.

## Saved results and verification

- `routes.png`: Routes before and after training.
- `policy_and_q_values.png`: The policy and Q table.
- `learning_curves.png`: Changes in rewards, exploration probability, and the starting state's Q value.
- `training.csv`: Per-episode data for creating additional teaching visualizations.
- `q_table.npy`: The final Q table.
- `q_snapshots.npz`: Q tables from selected training stages.
- `summary.json`: Run parameters, the final route, and whether the shortest path length was reached.

The program independently computes the shortest route length using BFS for verification; training does not use this answer.
This only verifies performance on a fixed map and starting position and does not demonstrate generalization to new maps.

## Classroom animations

```bash
python3 reinforcement-learning/animate_q_learning.py
```

Run from the repository root to generate:

- `results/q_learning_animation.gif`: A real training excerpt showing early exploration, the first time the goal is reached,
  episode 50, episode 1,500, and the final greedy evaluation. Transitions between excerpts are clearly marked.
- `results/q_learning_step_by_step.gif`: A slow playback of the last 8 steps before the goal is first reached, suitable for explaining the formula.
- `results/animation_trace.json`: The interaction trace used by the animation.

The left side shows the robot's actions and position; the right side displays the four Q values for every non-obstacle state.
Yellow marks the cell about to be updated, and green marks the updated cell; the numerical calculation appears at the bottom.
Displayed values are rounded to three decimal places, while internal calculations use full precision. The Q table is not updated during evaluation.
The final Q table is checked against the result saved by the original program, if available, and every displayed update is also checked.

Export an MP4 without a top-level title (1600×900, 20 fps, H.264):

```bash
python3 -m pip install imageio-ffmpeg
python3 reinforcement-learning/animate_q_learning.py --video
```

The output is `results/q_learning_animation_no_title.mp4`; it retains the stage, episode, step, and update information.

Export five separate widescreen videos (1600×780) without a top-level title, stage descriptions, or transition screens:

```bash
python3 reinforcement-learning/animate_q_learning.py --split-videos
```

The outputs are saved to `results/video_parts/` and numbered `01` through `05`: early exploration, first reaching the goal,
episode 50, episode 1,500, and final policy evaluation. They retain the step, epsilon, action, reward,
Q table, and update calculation. `manifest.json` records the training stage, duration, and video dimensions for each segment.

## Closing topic: From Q-learning to DQN

Tabular Q-learning stores one value for each state-action pair. For large or continuous state spaces,
a neural network can approximate the Q function, taking a state as input and outputting the Q values for each action.
DQN also uses experience replay and a target network to help stabilize training. This example focuses on the tabular method and does not implement DQN.
