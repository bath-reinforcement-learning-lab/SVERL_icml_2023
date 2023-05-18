import numpy as np

class Minesweeper:
    """
    RL env for Minesweeper. Edit from Scarllette Ellis's code.
    """

    def __init__(self, length, height, num_mines):

        self.length = length
        self.height = height
        self.num_mines = num_mines

        self.state_dim = self.height * self.length

        self.num_actions = self.state_dim
        self.actions = np.arange(self.num_actions)

        self.valid_dict = ValidDict()
        self.valid_dict.valid_actions = self.valid_actions
        self.valid_dict.actions = self.actions
        
        # For rendering
        self.render_dic = {-1 : 'B'}

    def reset(self, state=None):

        if state is None:

            self.board = np.full((self.height, self.length), -1)
            mine_inds = np.random.choice(np.arange(self.state_dim), self.num_mines, replace=False)
            mine_inds.sort()
            self.mine_locations = [(mine_ind % self.length, mine_ind // self.height) for mine_ind in mine_inds]

        else: # Sample from possible instances for a given state.

            full_state = self.instances[tuple(state)][0][np.random.choice(self.instances[tuple(state)][2], p=self.instances[tuple(state)][1])]
            self.board = full_state[:-2 * self.num_mines].reshape(self.height, self.length).astype(int)
            mine_locations = full_state[-2 * self.num_mines:].reshape(self.num_mines, 2)
            self.mine_locations = [tuple(mine_location) for mine_location in mine_locations]

        return self.board.flatten(), {'valid_actions':self.valid_dict[self.board.tobytes()]}

    def step(self, action):

        reward, done = self.dig(action)

        n_state = self.board.flatten()

        return n_state, reward, done, False, {'valid_actions':self.valid_dict[n_state.tobytes()]}

    def valid_actions(self, board):
        """
        Given a minesweeper board returns the valid actions.
        """

        return self.actions[board.flatten() == -1]
    
    def get_full_state(self):
        """
        Returns the MDP state and location of mines to completely specify board.
        """

        return np.append(self.board, self.mine_locations)

    def dig(self, action):

        y = action // self.height
        x = action % self.length

        done = False
        reward = 0

        if (x, y) in self.mine_locations: # Hit a mine
            self.board[y][x] = 9
            done = True
            reward -= 20

        else: # Guess is fine

            to_reveal = [(x, y)]
            num_to_reveal = 1
            while num_to_reveal > 0:
                cord = to_reveal.pop()
                num_to_reveal -= 1
                x = cord[0]
                y = cord[1]

                value = self.board[y][x]
                if value != -1:
                    continue

                value = self.find_value(x, y)
                self.board[y][x] = value

                if value == 0:
                    for j in range(y - 1, y + 2):
                        if j < 0 or j >= self.height:
                            continue
                        for i in range(x - 1, x + 2):
                            if (j == y and i == x) or i < 0 or i >= self.length:
                                continue
                            to_reveal.append((i, j))
                            num_to_reveal += 1

            if self.only_mines_left(): done = True # Game has been won.

        return reward, done

    def find_value(self, x, y):
        value = 0

        for j in range(y - 1, y + 2):
            if j < 0 or j >= self.height:
                continue
            for i in range(x - 1, x + 2):
                if (j == y and i == x) or i < 0 or i >= self.length:
                    continue
                value += int((i, j) in self.mine_locations)
        return value

    def only_mines_left(self):
        for y in range(self.height):
            for x in range(self.length):
                value = self.board[y][x]
                if value == -1 and (x, y) not in self.mine_locations:
                    return False

        return True

    def render(self):

        print()
        for row in self.board:
            to_print = ""
            for elm in row:
                to_print += " "
                if elm == -1: 
                    to_print += 'B'
                else:
                    to_print += str(elm)
            print(to_print)
        print()

class ValidDict(dict, Minesweeper):
    
    def __missing__(self, key):
        
        val = self.valid_actions(np.frombuffer(key, dtype=np.int_))
        self.__setitem__(key, val)
        
        return val