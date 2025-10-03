#######################################################################################
# Filename: HW2_Schafer.py
#
# Description: This script contains computations for AE 598 HW2.
#
# References:
#   Sutton&Barto.pdf: file:///C:/Users/wsche/Documents/AE%20598/Sutton&Barto.pdf
#
#######################################################################################
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
from plottingFuncs import *
from MCcontrol import *
from Sarsa import *
from QLearning import *
print('\n')

## User Inputs ##
is_slippery = True
num_episodes = 5000
gamma = 0.95
epsilon = 0.1
alpha = 0.1
num_evals = 50

see_MC = True
see_Sarsa = True
see_QL = True

## Monte Carlo Control on Frozen Lake MDP ##
if see_MC:
    pi_MC, pi2eval_MC, dt, returnsMC = MCcontrol(
                                          num_episodes=num_episodes, 
                                          gamma=gamma, 
                                          epsilon=epsilon,
                                          is_slippery=is_slippery, 
                                          max_steps=1000,
                                          num_evals=num_evals
                                         )
    risk_MC, risky_states_MC, hole_probs_MC = validatePi(pi_MC)
    print(f'Policy (MC): {pi_MC} \nRisk (MC): {risk_MC} \nRisky States (MC): {risky_states_MC} \nHole Probabilities (MC): {hole_probs_MC}\n')
    plotPolicy(pi_MC, title="Monte Carlo Control Policy Map")

## Sarsa on Frozen Lake MDP ##
if see_Sarsa:
    pi_sarsa, pi2eval_sarsa, dt = sarsa(
                              num_episodes=num_episodes, 
                              gamma=gamma, 
                              epsilon=epsilon,
                              alpha=alpha,
                              is_slippery=is_slippery, 
                              max_steps=1000,
                              num_evals=num_evals
                             )
    risk_sarsa, risky_states_sarsa, hole_probs_sarsa = validatePi(pi_sarsa)
    print(f'Policy (Sarsa): {pi_sarsa} \nRisk (Sarsa): {risk_sarsa} \nRisky States (Sarsa): {risky_states_sarsa} \nHole Probabilities (MC): {hole_probs_sarsa}\n')
    plotPolicy(pi_sarsa, title="Sarsa Policy Map")

## Q-Learning on Frozen Lake MDP ##
if see_QL:
    pi_QL, pi2eval_QL, dt = QLearning(
                               num_episodes=num_episodes, 
                               gamma=gamma, 
                               epsilon=epsilon,
                               alpha=alpha,
                               is_slippery=is_slippery, 
                               max_steps=1000,
                               num_evals=num_evals
                              )
    risk_QL, risky_states_QL, hole_probs_QL = validatePi(pi_QL)
    print(f'Policy (Q-Learning): {pi_QL} \nRisk (Q-Learning): {risk_QL} \nRisky States (Q-Learning): {risky_states_QL} \nHole Probabilities (MC): {hole_probs_QL}\n')
    plotPolicy(pi_QL, title="Q-Learning Policy Map")

if see_MC and see_Sarsa and see_QL:
    plotEvalReturn([pi2eval_MC, pi2eval_sarsa, pi2eval_QL], ['Monte Carlo Control', 'Sarsa', 'Q-Learning'], dt, gamma=1, eval_episodes=10000, is_slippery=is_slippery)
