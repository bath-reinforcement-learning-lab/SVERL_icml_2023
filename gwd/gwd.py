import numpy as np
import time
from collections import defaultdict
from itertools import product

class Grid:
    """
    W x H grid world with random goal.
    Grid has a boarder around the sides.
    A number of random grid squares have impassible blocks. 
    Reward is -1 each step and +10 for reaching the goal.
    Initial position is randomly sampled from any square.
    """

    def __init__(self, H, W, n_blocks):

        # Make grid and place blocks
        self.grid = np.zeros(H * W)
        self.block_inds = np.random.choice(H * W, n_blocks, replace=False)
        self.grid[self.block_inds] = 1
        self.grid = self.grid.reshape(H, W)

        # 4 available action
        self.num_actions = 4
        self.state_dim = 2

        # Initial states and goal
        self.not_blocks = [i for i in range(H * W) if i not in self.block_inds]
        self.goal_ind = np.random.choice(self.not_blocks)
        self.goal = np.append(self.goal_ind % H, self.goal_ind // H)
        self.not_blocks = np.array([i for i in self.not_blocks if i != self.goal_ind])
        self.start = np.append(self.not_blocks % H, self.not_blocks // H).reshape(2, len(self.not_blocks)).T

        # Transitions of grid world, for speedy look up.
        self.trans = self.trans = defaultdict(lambda: [[] for _ in range(4)])

        self.action_dict = {0: [0, 1],
                            1: [1, 0],
                            2: [0, -1],
                            3: [-1, 0]}
        
        # Transition for every state
        for state, a in product(self.start, range(4)):
                
            # Standard reward of -1 for step and episode not ended
            reward_done = np.array([-1, 0])

            # Converts action int to vector
            new_state = state.copy()
            new_state += self.action_dict[a]

            # Check if "hit wall" or goes to unavailable state
            if not ((new_state[1] in np.arange(H)) and (new_state[0] in np.arange(W))) or self.grid[new_state[1]][new_state[0]]: new_state = state
            if (new_state == self.goal).all(): reward_done += [20, 1] # Reached goal

            # Update transition vector
            self.trans[tuple(state)][a] = np.append(new_state, reward_done) 

    def reset(self, start_state=None):
        """
        Randomly places the agent in one of the bottom two squares.
        Or sets environment to given state.
        """

        if start_state is None: self.pos = self.start[np.random.choice(len(self.start))]
        else: self.pos = np.array(start_state)

        return self.pos.copy(), None

    def step(self, action):
        """
        Takes a step in the fully observed environment
        """

        new_state_reward_done = self.trans[tuple(self.pos)][action]
        self.pos = new_state_reward_done[:2]

        return self.pos.copy(), new_state_reward_done[2], new_state_reward_done[3], False, None