#######################################################################################
# Filename: plottingFuncs.py
#
# Description: This script contains functions for plotting metric relevant to policy
#              and value iterative methods for solving the 4x4 Frozen Lake MDP.
#
# Functions:
#   - plotPolicy()
#
#######################################################################################
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import re
from FrozenLakeMDP import *

## Plot the Policy Map of an Agent ##
def plotPolicy(pi, title, show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0,4)
    ax.set_ylim(0,4)
    ax.set_aspect("equal")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.grid(True, linewidth=1, alpha=0.4)

    X, Y, U, V = [], [], [], []

    # define environment and action space
    grid = ['SFFF', 'FHFH', 'FFFH', 'HFFG']
    action_space = {0:(-0.8, 0.0), 3:(0.0, -0.8), 2:(0.8, 0.0), 1:(0.0, 0.8)}

    # draw background cells and annotations
    for r in range(4):
        for c in range(4):
            ch = grid[r][c]

            # color by cell type
            if ch == "S":
                fc = "#5ac430"
            elif ch == "G":
                fc = "#2A5A17"
            elif ch == "H":
                fc = "#ff0000"
            else:
                fc = "#2ec0ff"
            ax.add_patch(plt.Rectangle((c, r), 1, 1, fc=fc, ec="none", alpha=0.9))

            # center of the cell and state index
            x, y = c + 0.5, r + 0.5
            s = r * 4 + c

            # draw arrows only on non-terminal cells
            if ch not in ("H", "G"):
                dx, dy = action_space[int(pi[s])]
                X.append(c + 0.5)
                Y.append(r + 0.5)
                U.append(dx)
                V.append(dy)

            # label H/G/S letters
            if ch in ("H", "G", "S"):
                if ch == "S":
                    ax.text(x, y-0.1, ch, ha="center", va="center", fontsize=14, fontweight="bold")
                else:
                    ax.text(x, y, ch, ha="center", va="center", fontsize=14, fontweight="bold")

    # plot the vector arrows
    ax.quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=1.0, pivot="mid", width=0.008)

    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(fname=f'{title}.png', bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax

## Plot the Trajectory of an Agent ##
def plotTrajectory(states, title, animate=False, fps=2):
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.82)
    ax.set_xlim(0,4)
    ax.set_ylim(0,4)
    ax.set_aspect("equal")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.grid(True, linewidth=1, alpha=0.4)

    # define environment
    grid = ['SFFF', 'FHFH', 'FFFH', 'HFFG']

    # draw background cells and annotations
    for r in range(4):
        for c in range(4):
            ch = grid[r][c]
            if ch == "S":
                fc = "#5ac430"
            elif ch == "G":
                fc = "#2A5A17"
            elif ch == "H":
                fc = "#ff0000"
            else:
                fc = "#2ec0ff"
            ax.add_patch(plt.Rectangle((c, r), 1, 1, fc=fc, ec="none", alpha=0.9))
            x, y = c + 0.5, r + 0.5
            if ch in ("S", "H", "G"):
                ax.text(x, y, ch, ha="center", va="center", fontsize=14, fontweight="bold")

    # convert state indices to cell centers
    xs, ys = [], []
    for s in states:
        r, c = divmod(s, 4)
        xs.append(c + 0.5)
        ys.append(r + 0.5)

    # draw trajectory
    ax.plot(xs, ys, marker="o", linewidth=2)
    if xs and ys:
        ax.scatter([xs[0]], [ys[0]], s=100, c="white", edgecolors="black", zorder=3)
        ax.scatter([xs[-1]], [ys[-1]], s=300, marker="*", c="gold", edgecolors="black", zorder=3)

    # side summary
    total_steps = max(len(states) - 1, 0)
    goal_reached = (states[-1] == 15) if states else False
    static_box = dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.3")
    static_text = ax.text(1.02, 0.05, f"Total Steps: {total_steps}\nGoal: {'Reached' if goal_reached else 'Not Reached'}", transform=ax.transAxes, ha="left", va="bottom", fontsize=11, bbox=static_box, clip_on=False)
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(fname=f'{title}.png', bbox_inches="tight")
    plt.show()

    # animate the trajectory as a GIF
    if animate:
        print(f'Animating {title}.gif ...')
        static_text.set_visible(False)
        line, = ax.plot([], [], marker="o", linewidth=2)
        head = ax.scatter([], [], s=300, marker="o", c="blue", edgecolors="black", zorder=4)

        N = len(xs)
        dyn_text = ax.text(1.02, 0.05, "", transform=ax.transAxes, ha="left", va="bottom", fontsize=11, bbox=static_box, clip_on=False)

        def init():
            line.set_data([], [])
            head.set_offsets(np.empty((0, 2)))
            dyn_text.set_text("")
            return line, head, dyn_text

        def update(i):
            line.set_data(xs[:i+1], ys[:i+1])
            head.set_offsets([[xs[i], ys[i]]])
            dyn_text.set_text(f"Step: {i+1}/{N}\nGoal: {'Reached' if goal_reached else 'Not Reached'}")
            return line, head, dyn_text

        anim = FuncAnimation(fig, update, init_func=init, frames=N, interval=int(1000 / fps), blit=True, repeat=False)

        gif_path = f"{title}.gif"
        writer = PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer, dpi=150)

        ax.set_title(title + f" (saved {gif_path})")
        ax.invert_yaxis()
        plt.show()

    return fig, ax

## Plot the Evaluation Return Vs. Time Steps ##
def plotEvalReturn(pi_sets, pi_names, dt, gamma=0.95, eval_episodes=50, map_name="4x4", is_slippery=True, max_steps=1000):

    # helper function
    def getReturns_all_visits(gamma, rewards, actions):
        T = len(actions)
        G_per_step = np.zeros(T, dtype=float)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + gamma * G
            G_per_step[t] = G
        return G_per_step

    fig = plt.figure()

    for alg_pis, name in zip(pi_sets, pi_names):
        y_vals = []
        for pi in alg_pis:
            ep_returns = []
            for _ in range(eval_episodes):
                states, actions, rewards, _ = simFrozenLakeMDP(pi=pi, map_name=map_name, is_slippery=is_slippery, max_steps=max_steps)
                G = getReturns_all_visits(gamma=gamma, rewards=rewards, actions=actions)
                ep_returns.append(float(np.nanmax(G)))
            y_vals.append(np.mean(ep_returns))

        x_vals = np.arange(len(alg_pis)) * int(dt)
        plt.plot(x_vals, y_vals, label=name, linewidth=2, marker='D', markersize=5)

    plt.xlabel("Time Steps")
    plt.ylabel(f"Evaluation Return (mean over {eval_episodes} episodes)")
    plt.title(f"Evaluation Return vs. Time Steps {pi_names}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.xlim(left=0)

    # make a safe filename from the title, e.g. "Eval_Return_vs_Steps.png"
    picName = f"Evaluation Return vs Steps {pi_names}"
    safe_name = re.sub(r"[^\w\-]+", "_", picName).strip("_") + ".png"
    fig.savefig(safe_name, dpi=200)

    return safe_name
