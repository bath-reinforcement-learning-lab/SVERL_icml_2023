import gym
import numpy as np

# Set environment and agent
class FactoredState(gym.ObservationWrapper):
    """
    Changes the state for taxi so that instead of discrete it is factored.
    """

    def __init__(self, env):

        super().__init__(env)

        self._observation_space = gym.spaces.MultiDiscrete([5, 5, 5, 4])

        self.state_dim = self.observation_space.shape[0]
        self.num_actions = self.action_space.n

    def decode(self, obs):

        # Decode function copied from Taxi-v3: https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py

        out = np.empty(4)
        out[3] = obs % 4; obs //= 4
        out[2] = obs % 5; obs //= 5
        out[1] = obs % 5; obs //= 5
        out[0] = obs
        return out

    def observation(self, obs):

        return self.decode(obs)
