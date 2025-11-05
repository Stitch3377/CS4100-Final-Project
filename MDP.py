import numpy as np

ACTIONS = np.array([
    [0,1],[1,0],[0,-1],[-1,0],#UP,DOWN,LEFT,RIGHT
    [-1,-1],[-1,1],[1,-1],[1,1],#Diagonals (DL, UL, DR, UR)

])

class MDP:
    """Class that represents a Markov Decision Process (MDP)"""

    def __init__(self, state_grid, delta_l, delta_w, cb):
        """DO NOT INITIALIZE THROUGH THIS; USE BUILDER"""
        self.state_grid = state_grid
        self.transition_matrix = MDP.states_to_transition(state_grid)
        self.delta_l = delta_l
        self.delta_w = delta_w
        self.cb = cb


    class MDP_builder:
        """Builder class for MDP class"""
        def __init__(self, cb, cl, cw, l_segments, w_segments):
            """Creates a builder object using vectors for three points and the amount of segments desired"""
            self.cb = cb
            self.delta_l = (cl - cb) / l_segments
            self.delta_w = (cw - cb) / w_segments
            self.grid = np.empty((l_segments, w_segments), dtype=bool)

        def feed_photo_loc(self, coord):
            """Feeds a photo's coordinates into the mdp; filling in that state if it did not exist before"""
            c = coord - self.cb
            pr_w = np.dot(c, self.delta_w) / np.linalg.norm(self.delta_w)
            pr_l = np.dot(c, self.delta_l) / np.linalg.norm(self.delta_l)
            x = int(pr_w)
            y = int(pr_l)
            self.grid[y, x] = 1

        def create(self):
            """Creates MDP object from current builder"""
            return MDP(self.grid, self.delta_l, self.delta_w, self.cb)

    #turns a state grid (which is a matrix of bools that represent the whether a grid piece is a state or not)
    #into a transition matrix
    def states_to_transition(state_grid):
        """turns a state grid (which is a matrix of bools that represent the whether a grid piece is a state or not)
            into a transition matrix"""
        num_states = np.sum(state_grid)
        transition_matrix = np.zeros((len(ACTIONS), num_states, num_states), dtype=bool)
        curr_state = 0
        for y in range(len(state_grid)):
            for x in range(len(state_grid[0])):
                if state_grid[y][x]== 0:
                    continue
                for action in range(len(ACTIONS)):
                    nx = x + ACTIONS[action][0]
                    ny = y + ACTIONS[action][1]
                    if nx < 0 or nx >= len(state_grid[0]) or ny < 0 or ny >= len(state_grid):
                        continue#out of bounds
                    if state_grid[ny][nx] == 0:
                        continue#no state at loc
                    new_state = np.sum(state_grid[:y])+np.sum(state_grid[y][:x])
                    transition_matrix[action][curr_state][new_state] = 1
        return transition_matrix

    def coord_to_state(self, coord):
        """Converts the latitude and latitude in a 2x1 np vector into the index of its state"""
        c = coord - self.cb
        pr_w = np.dot(c, self.delta_w) / np.linalg.norm(self.delta_w)
        pr_l = np.dot(c, self.delta_l) / np.linalg.norm(self.delta_l)
        x = int(pr_w)
        y = int(pr_l)
        return np.sum(self.state_grid[:y])+np.sum(self.state_grid[y][:x])

    def state_to_coord(self, state):
        """Converts a states index to its coordinates in latitude and longitude as a 2x1 np vector"""
        states_left = state
        for y in range(len(state)):
            for x in range(len(state[0])):
                states_left - self.state_grid[y][x]
                if states_left == 0:
                    return y * self.delta_l + x * self.delta_w


    def calc_transition_prob(self, old_state, action, new_state):
        """Calculates the T(s, a, s') value for a specific state, action, state tuple"""
        return self.transition_matrix[action][old_state][new_state]



