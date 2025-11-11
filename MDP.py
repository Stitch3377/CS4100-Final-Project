import numpy as np
from decimal import Decimal

ACTIONS = np.array([
    [0,1],[1,0],[0,-1],[-1,0],#UP,DOWN,LEFT,RIGHT
    [-1,-1],[-1,1],[1,-1],[1,1],#Diagonals (DL, UL, DR, UR)

])

class MDP:
    """Class that represents a Markov Decision Process (MDP)."""

    def __init__(self, state_grid, delta_l, delta_w, cb):
        """
        Private constructor for the MDP class, instead use builder to create objects.

        Parameters
        ----------
        state_grid : 2d numpy array with unknown shape
                     contains 1 if that space is a state, 0 otherwise.
        delta_l    : float
                     length of each grid space
        delta_w    : float
                     width of each grid space
        cb         : 2x1 numpy vector
                     coordinates of vertex being used as the new origin
        """
        self.state_grid = state_grid
        self.transition_matrix = MDP.__states_to_transition(state_grid)
        self.delta_l = delta_l
        self.delta_w = delta_w
        self.cb = cb


    class MDP_builder:
        """Builder class for MDP class."""
        def __init__(self, cb, cl, cw, l_segments, w_segments):
            """
            Creates a builder object using vectors for three points and the amount of segments desired.

            Parameters
            ----------
            cb         : 2x1 numpy vector
                         coordinate of vertex being used as the new origin
            cl         : 2x1 numpy vector
                         coordinate of vertex defining the length from the origin
            cw         : 2x1 numpy vector
                         coordinate of vertex defining the width from the origin
            l_segments : int
                         amount of segments to split length into
            w_segments : int
                         amount of segments to split width into
            """
            self.cb = cb
            self.delta_l = (cl - cb)/ l_segments
            self.delta_w = (cw - cb) / w_segments
            self.grid = np.zeros((l_segments, w_segments), dtype=bool)

        def feed_photo_loc(self, coord):
            """
            Feeds a photo's coordinates into the mdp; filling in that state if it did not exist before.

            Parameters
            ----------
            coord : 2x1 numpy vector
                    Coordinate of the photo to be fed into the mdp

            Returns
            -------
            MDP-Builder
                Current MDP-builder object
            """
            c = coord - self.cb
            norm_w = self.delta_w / np.dot(self.delta_w, self.delta_w)
            norm_l = self.delta_l / np.dot(self.delta_l, self.delta_l)
            pr_w = np.dot(c, norm_w)
            pr_l = np.dot(c, norm_l)
            x = int(pr_w)
            y = int(pr_l)
            try:
                self.grid[y, x] = 1
            except IndexError:
                print(f"out of bounds data: {coord}")
            return self

        def create(self):
            """
            Creates MDP object from current builder.

            Returns
            -------
            MDP
                MDP object created from the current builder
            """
            return MDP(self.grid, self.delta_l, self.delta_w, self.cb)

    def __states_to_transition(state_grid):
        """
        Turns a state grid (matrix of bools that represent if a grid piece is a state) into a transition matrix

        Parameters
        ----------
        state_grid : 2d numpy array with unknown shape
                     Matrix containing the states of the mdp with 1 being a grid having a state, 0 otherwise

        Returns
        -------
        3d numpy array
            Transition matrix formatted [action][state][new state]
        """
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
        """
        Converts the latitude and latitude in a 2x1 np vector into the index of its state.

        Parameters
        ----------
        coord : 2x1 numpy vector
                Coordinate of the point to be converted to a state's index

        Returns
        -------
        int
            The index of the state at that coordinate
        """
        c = coord - self.cb
        pr_w = np.dot(c, self.delta_w) / np.linalg.norm(self.delta_w)
        pr_l = np.dot(c, self.delta_l) / np.linalg.norm(self.delta_l)
        x = int(pr_w)
        y = int(pr_l)
        return np.sum(self.state_grid[:y])+np.sum(self.state_grid[y][:x])

    def state_to_coord(self, state):
        """
        Converts a states index to its coordinates in latitude and longitude as a 2x1 np vector.

        Parameters
        ----------
        state : int
                Index of the state to convert to the bottom left coordinate

        Returns
        -------
        2x1 numpy vector
            The coordinates of the bottom left vertex of that state
        """
        states_left = state
        for y in range(len(state)):
            for x in range(len(state[0])):
                states_left - self.state_grid[y][x]
                if states_left == 0:
                    return y * self.delta_l + x * self.delta_w


    def calc_transition_prob(self, old_state, action, new_state):
        """
        Calculates the T(s, a, s') value for a specific state, action, state tuple.

        Parameters
        ----------
        old_state : int
                    Index of the state starting at
        action    : int
                    Index of the action taken at that state
        new_state : int
                    Index of the new state reached by the action

        Returns
        -------
        float
            The probability of transitioning from old_state to new_state with the given action
        """
        return self.transition_matrix[action][old_state][new_state]

    def get_total_states(self):
        """
        Returns the total amount of states in the MDP.

        Returns
        int
            Number of states in the MDP
        -------

        """
        return np.sum(self.state_grid)