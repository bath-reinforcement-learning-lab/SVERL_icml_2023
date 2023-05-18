import numpy as np

class Grid:
    """
    2 x 4 grid world where top two squares are a goal.
    Grid has a boarder around the sides.
    (0, 1) is an impassible block.
    Reward is -1 each step and +10 for reaching the goal.
    Initial position is randomly sampled from bottom two squares.
    """

    def __init__(self):

        # 4 height and 2 width
        H = 4; W = 2
        self.grid = np.zeros((H, W))

        # 4 available action
        self.num_actions = 4
        self.state_dim = 2

        # Initial states
        self.start = np.array([[0, 0], [1, 0]])

        # Transitions of grid world, for speedy look up.
        # state, action then new_state + reward + done
        self.trans = np.zeros((W, H-1, 4, 4), dtype=int)

        self.action_dict = {0: [0, 1],
                            1: [1, 0],
                            2: [0, -1],
                            3: [-1, 0]}
        
        # Transition for every state
        for y, row in enumerate(self.grid[:-1]):
            for x, _ in enumerate(row):

                state = [x, y]

                # And every action
                for k in range(4):
                    
                    # Standard reward of -1 for step and episode not ended
                    reward_done = np.array([-1, 0])

                    # Converts action int to vector, 'Take step'
                    new_state = np.array(state) + self.action_dict[k]

                    # Check if "hit wall" or goes to unavailable state
                    if not ((new_state[1] in np.arange(H)) and (new_state[0] in np.arange(W))) or (new_state == [0, 1]).all(): new_state = state
                    if new_state[1] == H-1: reward_done += [10, 1] # Reached goal

                    # Update transition vector
                    self.trans[x, y, k] = np.append(new_state, reward_done) 

    def reset(self, start_state=None):
        """
        Randomly places the agent in one of the bottom two squares.
        Or sets environment to given state.
        """

        if start_state is None: self.pos = self.start[np.random.choice(2)]
        else: self.pos = np.array(start_state)

        return self.pos.copy(), None

    def step(self, action):
        """
        Takes a step in the fully observed environment
        """

        new_state_reward_done = self.trans[self.pos[0], self.pos[1], action]
        self.pos = new_state_reward_done[:2]

        return self.pos.copy(), new_state_reward_done[2], new_state_reward_done[3], False, None