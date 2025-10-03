#######################################################################################
# Filename: FrozenLakeMDP.py
#
# Description: This script contains functions for the Frozen Lake MDP environment.
#
# Functions:
#   - getFrozenLakePmatrix()
#   - simFrozenLakeMDP()
#   - validatePolicy()
#
# References:
#   Sutton&Barto.pdf: file:///C:/Users/wsche/Documents/AE%20598/Sutton&Barto.pdf
#
#######################################################################################
import numpy as np
import gymnasium as gym

## P Matrix Getter ##
def getFrozenLakePmatrix(map_name="4x4", is_slippery=True):
    ############################################################################
    # Function: getFrozenLakePmatrix
    #
    # Description: This function returns the Frozen Lake MDP transition
    #              probability matrix P(s'|s,a).
    #
    # Inputs:
    #   map_name:     string defining the grid size
    #   is_slippery:  boolean indicating if the ice is slippery
    #
    # Outputs: P
    #
    ############################################################################

    # instantiate the environment
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)

    # get P
    Pdict = env.unwrapped.P
    nS = env.observation_space.n
    nA = env.action_space.n
    P = [[list(Pdict[s][a]) for a in range(nA)] for s in range(nS)]
    env.reset()
    env.close()

    return P

## Frozen Lake MDP ##
def simFrozenLakeMDP(pi, map_name="4x4", is_slippery=True, max_steps=100):
    #################################################################################
    # Function: simFrozenLakeMDP
    #
    # Description: This function simulates a Frozen Lake MDP environment following a 
    #              defined policy. The environment consists of a 4x4 grid with start 
    #              (S), goal (G), frozen (F), and hole (H) states. The agent has an 
    #              action space (A) of left (0), down (1), right (2), and up (3). The
    #              agent is rewarded (+1) only when it reaches the goal state. If the
    #              ice is slippery, the agent has a 1/3 chance to move perpendicular 
    #              to the intended direction, but otherwise moves in the intended 
    #              direction.
    #
    # Inputs:
    #   pi:           policy (pi[s] = a)
    #   map_name:     string defining the grid size
    #   is_slippery:  boolean indicating if the ice is slippery
    #   max_steps:    maximum number of steps to simulate
    #
    # Outputs: 
    #   states:       sequence of states visited
    #   actions:      sequence of actions taken
    #   rewards:      sequence of rewards received
    #   goal_reached: boolean indicating if the goal was reached=
    #
    #################################################################################

    # initialize the gymnasium environment
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)
    s, _ = env.reset()

    # initialize MDP values
    if pi is None:
        pi = np.zeros(16, dtype=int)
    states = [int(s)]
    actions = []
    rewards = []
    goal_reached = False

    # run the sim
    for _ in range(max_steps):
        a = int(pi[s])
        s_next, r, terminated, truncated, _ = env.step(a)
        actions.append(a)
        rewards.append(float(r))
        states.append(int(s_next))

        # end if goal reached or failure
        if terminated or truncated:
            goal_reached = (r == 1.0)
            break

        s = s_next

    env.close()

    return states, actions, rewards, goal_reached

## Validate a Policy's Optimality ##
def validatePi(pi, map_name="4x4", is_slippery=True):
    #################################################################################
    # Function: validatePi
    #
    # Description: This function validates a given policy by checking and assessing
    #              the risk of every action within it.
    #
    # Inputs:
    #   pi:           policy (pi[s] = a)
    #   map_name:     string defining the grid size
    #   is_slippery:  boolean indicating if the ice is slippery
    #
    # Outputs: risk, risky_states, hole_prob
    #
    #################################################################################

    # initialize environment
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)
    P = env.unwrapped.P
    nS = env.observation_space.n
    desc = env.unwrapped.desc
    _, cols = desc.shape

    # identify holes and goals
    def cell_char(s):
        r, c = divmod(s, cols)
        return desc[r, c].decode("utf-8")
    holes = {s for s in range(nS) if cell_char(s) == "H"}
    goals = {s for s in range(nS) if cell_char(s) == "G"}
    terminals = holes | goals

    # perpendicular actions
    perps = {
        0: (1, 3),   # left: down, up
        1: (0, 2),   # down: left, right
        2: (1, 3),   # right: down, up
        3: (0, 2),   # up: left, right
    }

    # accumulate the risk
    risk = {}
    hole_prob = np.zeros(nS, dtype=float)

    for s in range(nS):
        if s in terminals:
            continue
        a = int(pi[s])

        p_hole = 0.0
        for prob, s_next, _, _ in P[s][a]:
            if s_next in holes:
                p_hole += prob

        hole_prob[s] = p_hole
        risk[s] = (p_hole > 0.0)

    env.close()
    risky_states = [s for s, v in risk.items() if v]
    hole_prob = np.round(hole_prob, 1)

    return risk, risky_states, hole_prob