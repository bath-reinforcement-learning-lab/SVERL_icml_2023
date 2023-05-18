import sys
sys.path.insert(0, '../')

from q_agent_2 import Agent
from tic_tac_toe import TTT
from utils import train, get_state_dist, F_not_i, tqdm_label
from characteristics import Characteristics
from shapley import Shapley
import numpy as np

env = TTT()
agent = Agent(env.state_dim, env.num_actions, epsilon=0.05, gamma=1, alpha=0.2)
states_to_explain = np.array([[0, 0, 0,
                               0, 1, 0,
                               2, 0, 2]]).astype(int)

# ------------------------------------------------- TRAIN
train(agent, env, 1e7)

# ------------------------------------------------- GET AGENT'S POLICY
agent.get_policy(env.valid_dict)

# ------------------------------------------------- APPROXIMATE STATE DIST
state_dist = get_state_dist(agent, env, 1e7)

# ------------------------------------------------- GET AGENT'S VALUE TABLE (for SHAP)
agent.get_value_table(env.valid_dict)

# ------------------------------------------------- ALL PI_C
pi_Cs = {tuple(C): agent.get_pi_C(C, state_dist, states_to_explain, env.valid_dict) for C in tqdm_label(F_not_i(np.arange(env.state_dim)), 'Calculating all pi_C')}

# ------------------------------------------------- ALL V_C
v_Cs = {tuple(C): agent.get_v_C(C, state_dist, states_to_explain) for C in tqdm_label(F_not_i(np.arange(env.state_dim)), 'Calculating all v_C')}

# ------------------------------------------------- ALL CHARACTERISTIC VALUES
characteristics = Characteristics(env, states_to_explain)
# Can do fast version because value function value function is the same everywhere (value function is deterministic)
local_sverl_characteristics = characteristics.fast_local_sverl_C_values(pi_Cs=pi_Cs, multi_process=True, num_p=25)
shapley_on_policy_characteristics = characteristics.shapley_on_policy(pi_Cs=pi_Cs, multi_process=True, num_p=25)
shapley_on_value_characteristics = characteristics.shapley_on_value(v_Cs=v_Cs, multi_process=True, num_p=25)

# ------------------------------------------------- SHAPLEY VALUES
shapley = Shapley(states_to_explain)
for characteristics, filename in zip([local_sverl_characteristics, 
                                      shapley_on_policy_characteristics, 
                                      shapley_on_value_characteristics], ['local', 'policy', 'value_function']):
    
    shapley_values = shapley.run(characteristics)
    print(shapley_values)

    import pickle
    with open('{}.pkl'.format(filename), 'wb') as file: pickle.dump(shapley_values, file)