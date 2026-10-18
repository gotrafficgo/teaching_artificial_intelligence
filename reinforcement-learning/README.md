# Q-learning: Learning to Navigate a Maze

一个面向大四学生的表格 Q-learning 例子。不需要数据集、神经网络或在线服务。

## Run the example

在仓库根目录运行（共享环境已包含依赖）：

```bash
python3 reinforcement-learning/q_learning.py
```

若缺少依赖：

```bash
python3 -m pip install -r reinforcement-learning/requirements.txt
```

默认训练 1500 轮，随机种子为 7；结果保存到本目录 `results/`。
运行时会打印一次手算示例和训练进度，并保存图片，无需图形窗口。

```bash
python3 reinforcement-learning/q_learning.py --episodes 3000 --seed 42
```

## Learning from an Example

机器人从 S 出发，绕过障碍 #，到达 G：

```text
S . . . .
. # # # .
. . . # .
. # . . .
. . . # G
```

- **State**：当前位置，`state = row * 5 + column`，行列从 0 开始。
- **Action**：上、右、下、左，分别编号 0、1、2、3。
- **Reward**：进入终点 +10，其他每步 −1；撞墙或障碍留在原位，也获得 −1。
- **Episode**：从起点开始，到达终点终止；最多收集 100 步，然后重置。
- **Q table**：25 × 4。障碍和终点的行不学习、不用于选择动作。
- **Policy**：根据 Q 表选择动作；训练时使用 ε-greedy 探索。

### Suggested demonstration order

1. 打开 `results/routes.png`，比较训练前后路线。
   训练前 Q 值全为 0，随机打破平局，因此行为随机。
   路线可能重叠，标题中的步数包含重复移动和撞墙。
2. 打开 `results/policy_and_q_values.png`，解释每格四个动作的评分与最佳动作。
   箭头来自估计的 Q 表；一次成功路线不代表每个位置的数值都完全收敛。
3. 结合终端中的两次手算，说明奖励信息如何向前传播。
4. 打开 `results/learning_curves.png`，比较带探索的训练和关闭探索的评估。
   评估不修改 Q 表，也不消耗训练用的随机数。最优动作平局时随机选择。

## Main Code in the Example

主算法在 `q_learning.py`，绘图单独放在 `plotting.py`。

| Function | Teaching focus |
| --- | --- |
| `step` | 状态转移与奖励是环境规则 |
| `choose_action` | ε-greedy 与随机打破平局 |
| `update_q` | 一次经历只更新一个 Q 值 |
| `train` | 反复交互、更新、重置 |
| `rollout` | 使用学到的策略，不再更新 |

核心更新：

```python
target = reward if terminated else reward + gamma * q[next_state].max()
q[state, action] += alpha * (target - q[state, action])
```

- `alpha = 0.2`：向新估计靠近多少。
- `gamma = 0.95`：未来奖励的折扣。
- `epsilon`：从 1.0 逐渐下降，最低保留 0.05 的探索概率。
- **终点与步数上限不同**：进入终点没有未来价值；100 步只是采样上限，
  更新仍保留下一状态价值，不把它误当作任务终点。

手算示例使用单独的表，不会预先向实际训练注入知识：

```text
Q(19, down) = 0 + 0.2 × (10 − 0) = 2.00
Q(14, down) = 0 + 0.2 × (−1 + 0.95 × 2 − 0) = 0.18
```

无折扣的每轮奖励总和用于绘图；Q 值估计的是折扣后的累积奖励，两者不同。
8 步到达终点的奖励总和是 `7 × (−1) + 10 = 3`。

## Saved results and verification

- `routes.png`：训练前后路线。
- `policy_and_q_values.png`：策略与 Q 表。
- `learning_curves.png`：回报、探索概率、起点 Q 值的变化。
- `training.csv`：逐轮数据，可用于后续教学图表。
- `q_table.npy`：最终 Q 表。
- `q_snapshots.npz`：若干训练阶段的 Q 表。
- `summary.json`：运行参数、最终路线、是否达到最短路径长度。

程序使用 BFS 独立计算最短路线长度进行核对；训练不使用这个答案。
这里只验证固定地图与固定起点上的表现，不表示能够泛化到新地图。

## Classroom animations

```bash
python3 reinforcement-learning/animate_q_learning.py
```

从仓库根目录执行，生成：

- `results/q_learning_animation.gif`：真实训练片段，展示早期探索、首次到达终点、
  第 50 轮、第 1500 轮和最后的贪心评估。片段之间明确标注跳转。
- `results/q_learning_step_by_step.gif`：慢速播放首次到达终点前的最后 8 步，适合讲公式。
- `results/animation_trace.json`：动画采用的交互记录。

左侧显示机器人动作和位置，右侧展示所有非障碍状态的四个 Q 值。
黄色表示即将更新的单元，绿色表示更新后的单元；底部展示数值计算。
数值显示保留三位小数，内部计算使用完整精度。评估阶段不更新 Q 表。
最终 Q 表会与原程序保存的结果核对（如果存在），所有展示的更新也会逐一核对。

导出无顶部主标题的 MP4（1600×900、20 fps、H.264）：

```bash
python3 -m pip install imageio-ffmpeg
python3 reinforcement-learning/animate_q_learning.py --video
```

输出为 `results/q_learning_animation_no_title.mp4`，保留阶段、episode、step 和更新信息。

导出五段独立的扁宽视频（1600×780），不带顶部标题、阶段说明和过场页：

```bash
python3 reinforcement-learning/animate_q_learning.py --split-videos
```

输出到 `results/video_parts/`，按 `01` 至 `05` 排列，依次为早期探索、首次到达终点、
第 50 轮训练、第 1500 轮训练和最终策略评估。保留 step、epsilon、动作、奖励、
Q 表及更新计算。`manifest.json` 记录每段对应的训练阶段、时长和视频尺寸。

## Closing topic: From Q-learning to DQN

表格 Q-learning 为每个状态和动作存储一个数值。状态空间很大或连续时，
可用神经网络近似 Q 函数：输入状态，输出各动作的 Q 值。
DQN 还使用经验回放与目标网络帮助稳定训练。本例聚焦表格方法，不实现 DQN。
