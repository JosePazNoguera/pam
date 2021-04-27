
import numpy as np
# from copy import deepcopy

class Scheduler:
    """
    Build a plan step by step.
    For each discrete time step, draw from a time-evolving probability distribution.
    """
    def __init__(self, person, planner):
        """
        """
        self.person = person
        self.planner = planner
        self.activity = 'home' # assume day starts at home
        self.area = person.home_area
        self.time = 0
        self.ongoing_duration = 0
        self.person_attributes = person.attributes
        self.valid_activities = ['home','work','shop','other']

    def __iter__(self):
        return self

    def __next__(self):
        """
        One-hour time step
        """

        # choose next activity
        transition_weights = self.planner.transition_matrix.loc(axis=0)[self.time, self.activity]
        next_activity = self.activity_choice(transition_weights)
        activity = self.activity if next_activity == 'no_change' else next_activity

        if activity != self.activity:
            self.ongoing_duration = 0
        self.activity = activity

        # time step
        self.time += 1
        if self.time > 23:
            raise StopIteration()

        return self

    def activity_choice(self, transition_weights):
        """
        :params pd.Series transition_weights: A pandas series with activities as index, and transition probabilities as values

        :returns str: next activity
        """
        # activity = np.random.choice(self.valid_activities)
        activity = transition_weights.sample(weights = transition_weights).index[0]
        return activity
        

    def location_choice(self):
        pass

    def mode_choice(self):
        pass