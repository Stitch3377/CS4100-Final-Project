import numpy as np
import src.mdp.MDP as MDP

class HMM:
    """Class that represents a Hidden Markov Model specifically only its belief state."""

    def __init__(self, MDP, CNN, first_observation):
        """
        Constructor for Hidden Markov Model.

        Parameters
        ----------
        MDP               : MDP.MDP
                            Markov Decision Process object used in the HMM, Should have same dimensions as CNN.
        CNN               : (Fill out in future)
                            Convolutional Neural Network model, Should have same dimensions as MDP.
        first_observation : (Fill out in future)
                            Photo that can be fed into the CNN. (Not yet known what format)
        """
        self.MDP = MDP
        self.CNN = CNN
        self.belief_state = None
        self.reset(first_observation)

    def step(self, action, observation):
        """
        Updates the Belief state of the HMM given a new action and observation.

        Parameters
        ----------
        action      : int
                      Integer id of the action that was taken.
        observation : (Fill out in future)
                      Photo that can be fed into the CNN. (Not yet known what format)
        """
        self.belief_state = np.matmul(self.belief_state, self.MDP.transition_matrix[action])
        #if CNN.get_observations returns col vector need to get transpose
        self.belief_state = self.belief_state * self.CNN.get_observations(observation)

    def reset(self, observation):
        """
        Resets the belief state of the HMM with a starting observation.

        Parameters
        ----------
        observation   : (Fill out in future)
                        Starting observation of the HMM which is a photo. (Not yet known what format)
        """
        self.belief_state = np.ones((self.MDP.get_total_states()))
        #if CNN.get_observations returns col vector need to get transpose
        self.belief_state = self.belief_state * self.CNN.get_observations(observation)

    def get_belief_state(self, state):
        """
        Returns the belief state of the HMM of a given state.

        Parameters
        ----------
        state : int
                Integer ID of the state to get the belief state for.

        Returns
        -------
        float
            Normalized probability of being in that state.
        """
        denominator = np.sum(self.belief_state)
        return self.belief_state[0][state]/denominator

    def belief_state_grid(self):
        """
        Returns the belief state grid of the HMM which is used for visualization purposes.

        Returns
        -------
        np.ndarray
            state grid which represents a map of northeastern. Instead all states are replaced with a float of how
            likely they are the correct state.
        """
        index = 0
        heat_map = np.zeros(self.MDP.state_grid.shape)
        for y in range(len(self.MDP.state_grid)):
            for x in range(len(self.MDP.state_grid[y])):
                if self.MDP.state_grid[y][x] == 0:
                    continue
                heat_map[y][x] = self.belief_state[index]
                index += 1
        return heat_map

