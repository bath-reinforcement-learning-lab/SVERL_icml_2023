import numpy as np
from collections import defaultdict
from multiprocessing import Manager, Process
import copy
from utils import F_not_i, tqdm_label

class Characteristics:
    """
    Calculates the characteristic values for different shapley values:
        - Local SVERL
        - Global SVERL
        - Shapley applied to value function
        - Shapley applied to policy
    """

    def __init__(self, env, states_to_explain, instances=None):

        self.env = env
        self.states_to_explain = states_to_explain
        self.instances = instances # Instances of environment for states being explained, only relevant for taxi.
        
        # For Shapley calculations
        self.F_card = env.state_dim
        self.F = np.arange(self.F_card)

        # If resetting is done by copying env or is built into reset function.
        if instances is None: self.reset_by_copy = False
        else: self.reset_by_copy = True

    def local_sverl_C_values(self, num_rolls, pi_Cs, multi_process=False, num_p=1):
        """
        Calculates local SVERL characteristics.
        
        num_rolls : Number of Monte Carlo roll outs for expected return.
        pi_Cs : All policies caused by removing every subset of features.
        """

        self.num_rolls = num_rolls
        self.pi_Cs = pi_Cs

        # Function for calculating partial policy for local SVERL
        self.get_policy = self.get_policy_local

        return self.get_all_C_values(self.get_local_global, multi_process, num_p)
    
    def fast_local_sverl_C_values(self, pi_Cs, num_rolls=1, valid_dict=None, multi_process=False, num_p=1):
        """
        Calculates local SVERL characteristics.
        Only valid for deterministic environments where states cannot be revisited.
        Much faster and more accurate.
        
        pi_Cs : All policies caused by removing every subset of features.
        """

        self.pi_Cs = pi_Cs

        # Do all the roll outs once now. Different values if available actions are state dependent or not.
        if valid_dict is None: self.action_values = {tuple(state) : np.mean([[self.play_episode(state.copy(), self.pi_Cs[tuple(self.F)], action) 
                                  for action in range(self.env.num_actions)] 
                                  for _ in tqdm_label(range(int(num_rolls)), 'Calculating Characteristics {}/{}'.format(i + 1, len(self.states_to_explain)))], axis=0)
                                  for i, state in enumerate(self.states_to_explain)}
            
        else: self.action_values = {tuple(state) : np.mean([[self.play_episode(state.copy(), self.pi_Cs[tuple(self.F)], action) if action in valid_dict[state.tobytes()] else 0 
                                  for action in range(self.env.num_actions)]
                                  for _ in tqdm_label(range(int(num_rolls)), 'Calculating Characteristics {}/{}'.format(i + 1, len(self.states_to_explain)))], axis=0)
                                  for i, state in enumerate(self.states_to_explain)}

        return self.get_all_C_values(self.get_fast_local, multi_process, num_p)

    def global_sverl_C_values(self, num_rolls, pi_Cs, multi_process=False, num_p=1):
        """
        Calculates global SVERL characteristics.
        Need pi_C for every state in env.
        
        num_rolls : Number of Monte Carlo roll outs for expected return.
        pi_Cs : All policies caused by removing every subset of features.
        """
        
        self.num_rolls = num_rolls
        self.pi_Cs = pi_Cs

        # Function for calculating partial policy for global SVERL
        self.get_policy = lambda state, C : copy.deepcopy(self.pi_Cs[tuple(C)]) 

        return self.get_all_C_values(self.get_local_global, multi_process, num_p)

    def shapley_on_policy(self, pi_Cs, multi_process=False, num_p=1):
        """
        Calculates Shapley applied to policy characteristics.
        
        pi_Cs : All policies caused by removing every subset of features.
        """

        self.pi_Cs = pi_Cs
        
        return self.get_all_C_values(self.get_shapley_on_policy, multi_process, num_p)

    def shapley_on_value(self, v_Cs, multi_process=False, num_p=1):
        """
        Calculates Shapley applied to value function characteristics.
        
        v_Cs : All expected return predictions caused by removing every subset of features.
        """

        self.v_Cs = v_Cs

        return self.get_all_C_values(self.get_shapley_on_value, multi_process, num_p)
        

# --------------------------------------------------------------------- Calculating a generic characteristic

    def get_all_C_values(self, get_C_values, multi_process=False, num_p=1):
        """
        Calculates all characteristic values for a given characteristic function.
        Multi or single processing. 
        """
        
        if multi_process: 

            characteristic_values = Manager().dict()
            all_C = F_not_i(self.F)

            for r in tqdm_label(range(int(np.ceil(len(all_C) / num_p))), 'Calculating Characteristics'):

                processes = [Process(target=self.worker, args=(C, characteristic_values, get_C_values)) for C in all_C[r * num_p : (r + 1) * num_p]]

                for p in processes:    
                    p.start()

                for p in processes:
                    p.join()

            return dict(characteristic_values)
            
        else: return {tuple(C): get_C_values(C) for C in tqdm_label(F_not_i(self.F), 'Calculating Characteristics')}
    
    def worker(self, C, characteristic_values, get_C_values): characteristic_values[tuple(C)] = get_C_values(C)

# --------------------------------------------------------------------- Local + Global SVERL 

    def get_local_global(self, C):
        """
        The SVERL characteristic values for one coalition for all states.
        play_policy dictates whether global or local is being calculated.
        """

        characteristic_values = defaultdict(float)

        for state in self.states_to_explain:

            play_policy = self.get_policy(state, C)

            for _ in range(int(self.num_rolls)): # Monte Carlo roll outs
                characteristic_values[tuple(state)] += self.play_episode(state.copy(), play_policy)

            characteristic_values[tuple(state)] /= self.num_rolls

        return dict(characteristic_values)
    
    def get_fast_local(self, C):
        """
        The local SVERL characteristic values for one coalition for all states.
        Only valid for deterministic environments where states cannot be revisited.
        Much faster and more accurate.

        V^{\pi_C}(s) = \sum_{a \in \A}{Q(s, a) * pi(a|s_C)}
        """

        return {tuple(state) : (value * self.pi_Cs[tuple(C)][tuple(state)]).sum() for state, value in self.action_values.items()}
    
    def get_policy_local(self, state, C):
        """
        Calculates the policy which local SVERL values uses for characteristic calculations.
        """

        play_policy = copy.deepcopy(self.pi_Cs[tuple(self.F)]) # Fully observed policy.
        play_policy[tuple(state)] = self.pi_Cs[tuple(C)][tuple(state)] # Partial policy for state being explained.

        return play_policy
    
    def play_episode(self, state, play_policy, action=None):
        """
        Plays an episode to evaluate a policy. (Monte Carlo roll out)
        """

        # Env set to state being explained, either using saved instance or built into env class.
        if self.reset_by_copy: self.env = copy.deepcopy(self.instances[tuple(state)])
        else: state, _ = self.env.reset(state)

        ret = 0

        if action is None: action = np.random.choice(self.env.num_actions, p=play_policy[tuple(state)])

        while True:

            # Usual RL, choose action, execute, update
            state, reward, terminated, truncated, _ = self.env.step(action)
            ret += reward

            if terminated or truncated: break
            else: action = np.random.choice(self.env.num_actions, p=play_policy[tuple(state)])

        return ret
    
# --------------------------------------------------------------------- Shapley applied to value function

    def get_shapley_on_value(self, C):
        """
        The Shapley applied to value function characteristic values for one coalition for all states.
        """

        return {tuple(state) : self.v_Cs[tuple(C)][tuple(state)] for state in self.states_to_explain}
    
# --------------------------------------------------------------------- Shapley applied to policy
    
    def get_shapley_on_policy(self, C):
        """
        The Shapley applied to policy characteristic values for one coalition for all states.
        """

        return {tuple(state) : self.pi_Cs[tuple(C)][tuple(state)] for state in self.states_to_explain}