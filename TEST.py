from cmath import e
from tkinter.tix import MAX
import numpy as np
import pandas as pd

np.random.seed(2)

N_STATES = 100  # number of goods in retailer inventory
ACTIONS = range(11)    # number of orders to factory per day
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 365*5   # maximum episodes
SELLINGPRICE = 5
CUST = np.random.randint(0, 10 + 1)
LEADTIME = 2
SAFETYSTOCK = 10   
ORDERCOST = SELLINGPRICE * 2
HOLDINGCOST = N_STATES / (N_STATES * SELLINGPRICE) * N_STATES
S_ = 0

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states + S_, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    #print(table)    # show table
    return table

def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, ]
    if (np.random.uniform() > EPSILON) or ((state_actions == 100).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name

def get_env_feedback(S, A):    # This is how agent will interact with the environment
    if A == 0:    # no order to factory
        if S < SAFETYSTOCK:   # current storage lower than 10
            R = -10
        else:
            if HOLDINGCOST <= CUST * SELLINGPRICE: 
                R = 1
            else:
                R = 0
    else:    # make order to factory
        S_ += A
        if HOLDINGCOST + ORDERCOST <= CUST * SELLINGPRICE:
            R = 1
        else:
            R = 0
    return R

def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        S = N_STATES
        is_terminated = False
        while not is_terminated:

            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)  # take action & get next state and reward
            q_predict = q_table.loc[S, A]
            if episode < MAX_EPISODES:
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
            else:
                is_terminated = True  # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)