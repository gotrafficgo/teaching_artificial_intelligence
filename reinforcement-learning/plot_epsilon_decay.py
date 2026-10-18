"""Export a slide-ready epsilon schedule using the course figure palette."""

from pathlib import Path

from plotting import plt, np


# Reuse these colors for subsequent teaching figures.
NAVY = "#082654"
BLUE = "#1463D6"
GREEN = "#087B3F"
PALE_BLUE = "#EAF4FD"
GRAY = "#53657B"


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15,
                         "text.color": NAVY, "axes.labelcolor": NAVY,
                         "xtick.color": GRAY, "ytick.color": GRAY,
                         "axes.spines.top": False, "axes.spines.right": False})
    episodes = np.arange(1, 1501)
    epsilon = np.maximum(0.05, 0.995 ** (episodes - 1))
    floor_episode = int(episodes[np.flatnonzero(epsilon == 0.05)[0]])

    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(.07, .90, "Epsilon Decay", fontsize=34, weight="bold")
    fig.text(.07, .845, "Gradually shift from exploration toward exploitation", fontsize=19, color=GRAY)
    ax = fig.add_axes((.085, .225, .59, .51))
    ax.fill_between(episodes, epsilon, color=PALE_BLUE)
    ax.plot(episodes, epsilon, color=BLUE, lw=3.8)
    ax.axhline(.05, color=GREEN, lw=1.6, linestyle=(0, (5, 4)))
    ax.scatter([1, floor_episode], [1, .05], color=[BLUE, GREEN], s=80, zorder=5, clip_on=False)
    ax.annotate("Start: 1.00", xy=(1, 1), xytext=(105, .94), color=BLUE, fontsize=16, weight="bold")
    ax.annotate(f"Minimum reached at episode {floor_episode}", xy=(floor_episode, .05),
                xytext=(700, .34), fontsize=14, color=GREEN,
                arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.5},
                va="center")
    ax.set(xlim=(0, 1500), ylim=(0, 1.08), xlabel="Episode", ylabel="Exploration probability, ε",
           xticks=[0, 300, 600, 900, 1200, 1500], yticks=[0, .2, .4, .6, .8, 1])
    ax.grid(axis="y", color="#E2E9F1", linewidth=.8)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#BAC9D9")
    ax.tick_params(length=0, pad=10)
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15

    fig.text(.735, .70, "OUR EXAMPLE", fontsize=14, weight="bold", color=GRAY)
    for y, label, value, color in [(.60, "Initial ε", "1.00", BLUE),
                                    (.46, "Decay rate", "0.995", BLUE),
                                    (.32, "Minimum ε", "0.05", GREEN)]:
        fig.text(.735, y+.055, label, fontsize=15, color=GRAY)
        fig.text(.735, y, value, fontsize=28, weight="bold", color=color)
    fig.text(.5, .115, r"$\varepsilon_{k+1}=\max(0.05,\;0.995\,\varepsilon_k)$",
             fontsize=23, ha="center")
    fig.text(.5, .055, "Fixed within each episode  •  Reduced between episodes", fontsize=16, ha="center", color=GRAY)
    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(output / f"epsilon_decay.{extension}", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Minimum ε first reached at episode {floor_episode}. Figures saved to {output}.")


if __name__ == "__main__":
    main()
