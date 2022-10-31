from cmath import e
from tkinter.tix import MAX
import numpy as np
import pandas as pd

np.random.seed(2)

N_STATES = list(range(201))
ACTIONS = list(range(11))
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
S = 100
MAX_EPISODES = 365
CUST = np.random.randint(0, 10 + 1)
LEADTIME = 2
SAFETYSTOCK = 10   
ORDERCOST = 3
HOLDINGCOST = 5 / 365 * S
S_ = 0

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((len(n_states), len(actions))),
        columns=actions,
    )
    #print(table)    # show table
    return table

def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name

def get_env_feedback(S, A):
    if A == 0:

        if S < SAFETYSTOCK:
            R = -10
            S_ = S - CUST

        else:
            R = HOLDINGCOST * -1
            S_ = S - CUST

            else:

        if S < SAFETYSTOCK:
            R = -10
            S_ = S - CUST + A

        else:
            R = (HOLDINGCOST + ORDERCOST) * -1
            S_ = S - CUST + A

    return S_, R

def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        S = 100
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