#######################################################################################
# Filename: Q-Learning.py
#
# Description: This script contains functions for performing the Q-Learning learning
#              method on a given MDP.
#
# Functions:
#   - getActionFromQ()
#   - TD0_Qest()
#   - Qlearning()
#
# References:
#   Sutton&Barto.pdf: file:///C:/Users/wsche/Documents/AE%20598/Sutton&Barto.pdf
#
#######################################################################################
import numpy as np
import gymnasium as gym
import time
from FrozenLakeMDP import *

## Greedy Action from Q ##
def getActionFromQ(Q, s, epsilon):
    ############################################################################
    # Function: getActionFromQ
    #
    # Description: This function computes the greedy action from a given action
    #              value function Q at a given state.
    #
    # Inputs:
    #   Q:       action value function
    #   s:       state
    #   epsilon: probability of taking an action other than max(Q(s))
    #
    # Outputs: a
    #
    ############################################################################

    # initialize values
    nA = Q.shape[1]

    # compute greedy action
    actionVals = Q[s]
    maxA = actionVals.max()
    best = np.flatnonzero(actionVals == maxA)

    # determine if action is randomized
    if np.random.rand() < epsilon:
        a = np.random.randint(nA)
    else:
        a = int(np.random.choice(best))

    return a

## TD(0) Q Estimation ##
def TD0_Qest_max(Q, s0, a0, s1, r, alpha, gamma=0.95, terminal=False):
    ############################################################################
    # Function: TD0
    #
    # Description: This function computes the policy from the action value 
    #              function based on the one-step temporal difference learning 
    #              method.
    #
    # Inputs:
    #   Q:     action value function
    #   s0:    previous state
    #   a0:    previous action
    #   s1:    new state
    #   r:     reward at new state
    #   alpha: learning rate
    #   gamma: discount factor
    #
    # Outputs: pi
    #
    ############################################################################

    # compute the Q update
    if terminal:
        target = r
    else:
        a1 = getActionFromQ(Q, s1, epsilon=0)
        target = r + gamma * Q[s1, a1]

    Q[s0, a0] += alpha * (target - Q[s0, a0])

    return Q

## TD0 Policy Estimation ##
def QLearning(num_episodes, gamma=0.95, epsilon=0.1, alpha=0.1, map_name="4x4", is_slippery=True, max_steps=100, num_evals=500):
    ############################################################################
    # Function: QLearning
    #
    # Description: This function computes the optimal policy by updating the
    #              action value function at every step, generating a new policy
    #              upon termination targeting a greedy action.
    #
    # Inputs:
    #   num_episodes: number of episodes
    #   gamma:        discount factor
    #   epsilon:      probability of taking an action other than max(Q(s))
    #   alpha:        learning rate
    #   map_name:     string defining the grid size
    #   is_slippery:  boolean indicating if the ice is slippery
    #   max_steps:    maximum number of steps to simulate
    #   num_evals:    number of policies collected for post-evaluation
    #
    # Outputs: pi, episodeReturns
    #
    ############################################################################

    # initialize values
    tStart = time.perf_counter()
    Q = np.zeros((16,4), dtype=float)
    dt = int(np.ceil(num_episodes / num_evals))
    eval_num = 0
    pi2eval = []
    numSteps = 0

    # run the MDP
    for _ in range(num_episodes):

        # initialize the gymnasium environment
        env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)
        s, _ = env.reset()

        # compute a at s from Q (every episode has a chance of randomness epsilon)
        a = getActionFromQ(Q, s, epsilon)
        
        # run the sim
        for _ in range(max_steps):

            # choose a from pi derived from Q
            s_next, r, terminated, truncated, _ = env.step(a)
            numSteps += 1

            # update Q
            a_next = getActionFromQ(Q, s_next, epsilon)
            Q = TD0_Qest_max(Q, s, a, s_next, r, alpha, gamma, terminated)

            # end if goal reached or failure
            if terminated or truncated:
                break

            s = s_next
            a = a_next

        env.close()

        # update and/or collect the policy for post evaluation
        if (eval_num == dt) or (eval_num == 0):
            pi_collect = np.argmax(Q, axis=1)
            pi2eval.append([numSteps, pi_collect])
            eval_num = 0
        eval_num += 1
    
    # compute final pi
    pi = np.argmax(Q, axis=1)

    # end the timer
    tEnd = time.perf_counter()
    elapsed = tEnd - tStart
    print(f'[Q-Learning] Elapsed {elapsed:.3f} seconds.')

    return pi, pi2eval, dt