from tqdm import tqdm
import copy
from collections import defaultdict
import numpy as np

def train(agent, env, num_steps):
    """
    Trains an agent for a set number of steps.
    """

    state, info = env.reset()

    for _ in tqdm_label(range(int(num_steps)), 'Training Agent'):

        # Usual RL, choose action, execute, update
        action = agent.choose_action(state, info)
        new_state, reward, terminated, truncated, info = env.step(action)
        agent.update(state, action, new_state, reward, terminated or truncated, info)
        state = new_state

        if terminated or truncated: state, info = env.reset()

def find_states_taxi(agent, env, states_to_explain):
    """
    Saves instances of the environment for states to explain.
    Only used for TAXI.
    """

    instances = {}

    state, info = env.reset()

    while len(states_to_explain) > len(instances):

        # Saving instance of the environment for state which is being explained.
        if (state == states_to_explain).all(axis=1).any(): 
            if tuple(state) not in instances: instances[tuple(state)] = copy.deepcopy(env)

        # Usual RL, choose action, execute, update
        state, _, terminated, truncated, info = env.step(agent.choose_action(state, info))

        if terminated or truncated: state, info = env.reset()

    return instances

def find_states_minesweeper(agent, env, states_to_explain, num_steps):
    """
    Saves possible instances of environment for states to explain.
    A state could have many instances, e.g., same state with different possible mine locations in minesweeper.
    Only used for minesweeper.
    """

    instances = defaultdict(lambda: defaultdict(float))

    state, info = env.reset()

    for _ in tqdm_label(range(int(num_steps)), 'Finding States'):

        # Saving instance of the environment for state which is being explained.
        if (state == states_to_explain).all(axis=1).any(): instances[tuple(state)][tuple(env.get_full_state())] += 1

        # Usual RL, choose action, execute, update
        state, _, terminated, truncated, info = env.step(agent.choose_action(state, info))

        if terminated or truncated: state, info = env.reset()

    # Possible states, probabilities of seeing those states and number of possible states.
    env.instances = {state: [np.array(list(value.keys())), 
                             np.array(list(value.values())) / sum(list(value.values())), 
                             len(np.array(list(value.keys())))] for state, value in instances.items()}

def get_state_dist(agent, env, sample_size):
    """
    Approximates the limiting state distribution.
    """

    state_dist = defaultdict(float)

    for state in agent.policy: # Populate the state distribution with all states in policy.
        state_dist[state]

    state, _ = env.reset()

    for _ in tqdm_label(range(int(sample_size)), 'Approximating State Distribution'):

        state_dist[tuple(state)] += 1
        agent.Q_table[tuple(state)] # To keep number of states in state dist and Q table the same.
        state, _, terminated, truncated, _ = env.step(np.random.choice(env.num_actions, p=agent.policy[tuple(state)]))

        if terminated or truncated: state, _ = env.reset()

    for state in state_dist:

        state_dist[state] /= int(sample_size)

    return state_dist

def F_not_i(F, feature=-1):
    """
    Finds all subsets not containing a feature.
    Given no feature, it returns all subsets of F.
    """ 

    all_C = []
    F_card = len(F)

    for i in range(1 << F_card):
        pos = [F[j] for j in F if (i & (1 << j))]
        if feature not in pos:
            all_C.append(pos) 

    return all_C

def mask_state(state, state_dim, C):
    """
    Takes a state and masks out state features according to a coalition.
    """

    not_C = [i for i in range(state_dim) if i not in C]

    out = state.copy()
    out[..., not_C] = -1

    return out

def value_iteration(env, gamma):
    """
    Performs value iteration, to get exact results faster in environments.
    Used for Taxi.
    """

    # Get transition matrix from env
    P = env.P
    num_states = len(P)
    V = np.zeros(num_states)

    # Perform value iteration
    delta = float('inf')
    terminal_states = [P[s][a][0][1] for s in range(num_states) for a in range(env.num_actions) if P[s][a][0][3]]            
    while delta > 1e-10:
        delta = 0.
        # Iterate over all states
        for s in range(num_states):
            if s not in terminal_states:
                v_prev = V[s]
                V[s] = np.max([P[s][a][0][2] + gamma*V[P[s][a][0][1]] for a in range(env.num_actions)])
                delta = max(delta, abs(v_prev-V[s]))

    # Get policy and Q table.
    policy = defaultdict(lambda: np.full(env.num_actions, 1/env.num_actions))
    Q_table = defaultdict(lambda: np.zeros(env.num_actions))
    for s in range(num_states):
        if s not in terminal_states:
            state = env.decode(s)
            Q_table[tuple(state)] = np.array([P[s][a][0][2]+gamma*V[P[s][a][0][1]] for a in range(env.num_actions)])
            policy[tuple(state)] = (Q_table[tuple(state)] == Q_table[tuple(state)].max()).astype(float)
            policy[tuple(state)] /= policy[tuple(state)].sum()

    return Q_table, policy

def tqdm_label(iterator, label):
    """
    Takes an iterator and produces a labelled tqdm progress bar.
    """

    pbar = tqdm(iterator)
    pbar.set_description(label)
    return pbar