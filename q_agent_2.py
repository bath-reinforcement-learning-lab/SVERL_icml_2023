import numpy as np
from collections import defaultdict
from utils import mask_state
import copy
    
class Agent:
    """
    Usual Q learning agent, with added functionality:
        - for calculting Shapley values.
        - for environments where action space is state dependent.
    """
    
    def __init__(self, state_dim, num_actions, epsilon=0.05, gamma=0.99, alpha=0.2):
        
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.actions = np.arange(self.num_actions)

        # Agent hyper-parameters
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        
        self.Q_table = defaultdict(lambda: np.zeros(self.num_actions))

    def choose_action(self, state, info, exp=True):
        """
        Chooses action with epsilon greedy policy.
        Info contains which actions are available in the state.
        """

        if np.random.rand() < self.epsilon and exp: return np.random.choice(info['valid_actions'])
        else: 
            q_values = self.Q_table[tuple(state)][info['valid_actions']]
            return np.random.choice(info['valid_actions'][q_values == q_values.max()])
        
    def update(self, state, action, new_state, reward, done, info):
        """
        Q learning update. Only look ahead over available actions.
        """

        # Clause to stop error when valid "actions" is empty at end of episode.
        if done: q_max = 0
        else: q_max = self.Q_table[tuple(new_state)][info['valid_actions']].max()
        
        # Usual update, for only valid "actions"
        td_error = reward + self.gamma * q_max - self.Q_table[tuple(state)][action]
        self.Q_table[tuple(state)][action] += self.alpha * td_error

# ------------------------------------------------------------------ Usual agent stuff finishes.

    def get_policy(self, valid_dict):
        """
        Converts the agent's Q table into a policy table.
        """

        # Slightly more complicated policy as probability of choosing unavailable actions has to be zero.
        self.policy = PolicyDict()
        self.policy.valid_dict = valid_dict
        self.policy.num_actions = self.num_actions

        for state, q_values in self.Q_table.items():

            # Q values with slightly different values but should be same policy. 
            actions = valid_dict[np.array(state).tobytes()]
            q_values = q_values[actions].round(2)

            self.policy[state][actions] = q_values == q_values.max()
            self.policy[state] /= self.policy[state].sum()

    def get_value_table(self, valid_dict):
        """
        Converts the agent's Q table into a value table.
        """

        self.value_table = {state: q_values[valid_dict[np.array(state).tobytes()]].max() for state, q_values in self.Q_table.items()}

    def get_pi_C(self, C, state_dist, states_to_explain, valid_dict):
        """
        Calculates pi_C for given states.
        """

        if len(C) == self.state_dim: return copy.deepcopy(self.policy)
        else:

            # Mask out features not in C to find states which share observations.
            all_states = np.array(list(self.policy.keys()))
            mask_states = mask_state(states_to_explain, self.state_dim, C)
            mask_all_states = mask_state(all_states, self.state_dim, C)

            # Limiting state occupancy distribution: p^{\pi}(s)
            state_dist_full = np.array(list(state_dist.values())) + 1e-16 # Jitter for divide 0

            # For making new policy
            pi_C = PolicyDict()
            pi_C.valid_dict = valid_dict
            pi_C.num_actions = self.num_actions
            temp_pi_C = {}

            for m_state in np.unique(mask_states, axis=0):

                ind = (mask_all_states == m_state).all(axis=1)
                state_dist_cond = state_dist_full[ind] / state_dist_full[ind].sum() # Conditional limiting state occupancy distribution.

                # pi_C = \sum_{s \in S}{\pi(a|s) * p(s|s_C)}
                temp_pi_C[tuple(m_state)] = (np.array(list(self.policy.values()))[ind] * state_dist_cond[:, None]).sum(axis=0)

            # Set partially observed policies for fully observed states using^
            for state, m_state in zip(states_to_explain, mask_states):

                # Setting impossible action probs to zero and renormalising
                actions = valid_dict[state.tobytes()]
                pi_C[tuple(state)][actions] = temp_pi_C[tuple(m_state)][actions]
                pi_C[tuple(state)] /= pi_C[tuple(state)].sum()
                
            return pi_C

    def get_v_C(self, C, state_dist, states_to_explain):
        """
        Converts the q_values for SHAP
        """

        # Mask out features not in C to find states which share observations.
        all_states = np.array(list(self.policy.keys()))
        mask_states = mask_state(states_to_explain, self.state_dim, C)
        mask_all_states = mask_state(all_states, self.state_dim, C)

        # Limiting state occupancy distribution: p^{\pi}(s)
        state_dist_full = np.array(list(state_dist.values())) + 1e-16 # Jitter for divide 0

        # For making new value table
        v_C = {}
        temp_v_C = {}
        values = np.array(list(self.value_table.values()))

        for m_state in np.unique(mask_states, axis=0):

            ind = (mask_all_states == m_state).all(axis=1)
            state_dist_cond = state_dist_full[ind] / state_dist_full[ind].sum() # Conditional limiting state occupancy distribution.
            
            # v_C = \sum_{s \in S}{V(s) * p(s|s_C)}
            temp_v_C[tuple(m_state)] = (values[ind] * state_dist_cond).sum() 

        for state, m_state in zip(states_to_explain, mask_states):

            v_C[tuple(state)] = temp_v_C[tuple(m_state)]

        return v_C

class PolicyDict(dict):
    """
    Special policy dictionary which sets the initial policy to random between available actions.
    """
    
    def __missing__(self, key):

        valid_actions = self.valid_dict[np.array(key).tobytes()]
        val = np.zeros(self.num_actions)
        val[valid_actions] = 1 / len(valid_actions)


        self.__setitem__(key, val)
        
        return val