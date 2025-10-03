#######################################################################################
# Filename: MCcontrol.py
#
# Description: This script contains functions for performing Monte Carlo Control on a
#              given MDP.
#
# Functions:
#   - getPiFromQ()
#   - getFirstVisitReturns()
#   - MCcontrol()
#
# References:
#   Sutton&Barto.pdf: file:///C:/Users/wsche/Documents/AE%20598/Sutton&Barto.pdf
#
#######################################################################################
import numpy as np
import time
from FrozenLakeMDP import *

## Greedy Policy from Q ##
def getPiFromQ(Q, epsilon):
    ############################################################################
    # Function: getPiFromQ
    #
    # Description: This function computes the greedy policy from a given action
    #              value function Q.
    #
    # Inputs:
    #   Q:       action value function
    #   epsilon: probability of taking an action other than max(Q(s))
    #
    # Outputs: pi
    #
    ############################################################################

    # initialize values
    nS, nA = Q.shape
    pi = np.zeros(nS, dtype=int)

    # generate a greedy policy
    for s in range(nS):

        # determine the action with the highest value (break ties randomly)
        actionVals = Q[s]
        max_action = actionVals.max()
        best = np.flatnonzero(actionVals == max_action)
        greedy_a = int(np.random.choice(best))

        # exploring starts (consider epsilon)
        if np.random.rand() < epsilon:
            pi[s] = np.random.randint(nA)
        else:
            pi[s] = greedy_a

    return pi

## First Visit Returns ##
def getFirstVisitReturns(gamma, rewards, states, actions):
    ############################################################################
    # Function: getFirstVisitReturns
    #
    # Description: This function computes the first-visit returns for every
    #              state-action pair in an episode.
    #
    # Inputs:
    #   gamma:   discount factor
    #   rewards: chronological list of rewards earned
    #   states:  chronological list of states visited
    #   actions: chronological list of actions taken
    #
    # Outputs: G
    #
    ############################################################################

    # initialize values
    G = np.full((16,4), np.nan, dtype=float)
    visited = set()
    
    # compute Gt for all (s,a)
    Gt = 0
    for step in reversed(range(len(actions))):
        Gt = rewards[step] + gamma * Gt
        s, a = states[step], actions[step]
        if (s,a) not in visited:
            visited.add((s,a))
            G[s,a] = Gt

    return G

## Monte Carlo Estimation ##
def MCcontrol(num_episodes, gamma=0.95, epsilon=0.1, map_name="4x4", is_slippery=True, max_steps=100, num_evals=500):
    ############################################################################
    # Function: MCestimation
    #
    # Description: This function computes the near-optimal policy by averaging
    #              the returns over a number of episodes.
    #
    # Inputs:
    #   num_episodes: number of episodes
    #   gamma:        discount factor
    #   epsilon:      probability of taking an action other than max(Q(s))
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

    # run the MDP and collect each episode's return
    episodeReturns = []
    for _ in range(num_episodes):

        # compute pi from Q (every episode has a chance of randomness epsilon)
        pi = getPiFromQ(Q, epsilon)

        # simulate the MDP
        states, actions, rewards, _ = simFrozenLakeMDP(pi=pi, 
                                                       map_name=map_name, 
                                                       is_slippery=is_slippery, 
                                                       max_steps=max_steps, 
                                                       )
        
        # compute the episode return
        Ge = getFirstVisitReturns(gamma=gamma, rewards=rewards, states=states, actions=actions)
        episodeReturns.append(Ge)
    
        # compute Q=average(G) and replace all NaN values with Q_init's values
        Qnew = np.nanmean(np.stack(episodeReturns, axis=0), axis=0)
        Qnew[np.isnan(Qnew)] = Q[np.isnan(Qnew)]

        # update loop
        Q = Qnew

        # update and/or collect the policy for post evaluation
        if (eval_num == dt) or (eval_num == 0):
            pi2eval.append(np.argmax(Q, axis=1))
            eval_num = 0
        eval_num += 1

    # compute pi
    pi = np.argmax(Q, axis=1)

    # end the timer
    tEnd = time.perf_counter()
    elapsed = tEnd - tStart
    print(f'[MC Method] Elapsed {elapsed:.3f} seconds.')

    return pi, pi2eval, dt, episodeReturns
