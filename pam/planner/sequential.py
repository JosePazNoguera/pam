
import numpy as np
import random
from pam import variables
from pam.core import Person
from pam.activity import Plan, Activity, Leg
from pam.utils import minutes_to_datetime
# from copy import deepcopy

class Scheduler:
    """
    Build daily plan
    """
    def __init__(self, person, planner):
        sequencer = Sequencer(person, planner)
        self.schedule = [self.record_values(sequencer)]
        self.person = person
        for step in sequencer:
            self.schedule.append(self.record_values(step))

        # add PAM plans
        self.add_plans()

    def record_values(self, sequencer):
        record_vars = ['time','activity','area','ongoing_duration']
        return {key:value for key, value in sequencer.__dict__.items() if key in record_vars}

    def add_plans(self):
        self.person.clear_plan() # empty any existing plans
        activity = self.schedule[0]['activity']
        start_time = self.schedule[0]['time']
        area = self.schedule[0]['area']
        seq = 1
        for step in self.schedule:
            if step['activity'] != activity or step['time']==23:
                end_time = step['time']+1
                self.person.add(Activity(seq=seq, act=activity, area=area,
                                start_time=minutes_to_datetime(start_time * 60),
                                end_time=minutes_to_datetime(end_time * 60)
                                )
                )

                #TODO: add legs
                self.person.add(Leg(seq=seq,mode='car',start_area='a',end_area='b',
                        start_time=minutes_to_datetime(end_time * 60),
                        end_time=minutes_to_datetime(end_time * 60)))

                area = step['area']
                start_time = step['time']
                activity = step['activity']
                seq += 1

class Sequencer:
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


        # transition_weights = self.planner.transition_matrix.loc(axis=0)[self.time, self.activity]
        # next_activity = self.activity_choice(transition_weights)
        # activity = self.activity if next_activity == 'no_change' else next_activity

        # if activity != self.activity:
        #     self.ongoing_duration = -1
        # self.activity = activity

        # switch activity?
        switch_activity = self.switch_activity_choice(self.planner.duration_pdf, self.activity, self.ongoing_duration+1)

        # if yes choose next activity
        if switch_activity or self.activity == 'home':      

            transition_weights = self.planner.transition_matrix.loc(axis=0)[self.time, self.activity]

            if self.activity != 'home':                
                # exclude "no change", since we have already decided to switch activity
                # TODO: improve this
                transition_weights = transition_weights[transition_weights.index!=self.activity]
                transition_weights = transition_weights.replace(0, variables.SMALL_VALUE)
                transition_weights = transition_weights / transition_weights.sum() # weights should add up to one

            next_activity = self.activity_choice(transition_weights)
            activity = self.activity if next_activity == 'no_change' else next_activity

            if activity != self.activity:
                self.ongoing_duration = -1
            self.activity = activity

        # time step
        self.time += 1
        self.ongoing_duration += 1
        if self.time > 23:
            raise StopIteration()

        return self

    def switch_activity_choice(self, pdf, act, ongoing_duration):
        """
        Given the ongoing duration of the activity, decide whether to switch to a new activity or not
        :returns bool: True -> switch activity, False -> don't switch
        """
        return random.random() < pdf[act](ongoing_duration)[0]+0.1 #TODO: remove this hardcoded uplift 

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