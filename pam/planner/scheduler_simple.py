
import numpy as np
import random
from pam import variables
from pam.core import Person
from pam.activity import Plan, Activity, Leg, Location
from pam.utils import minutes_to_datetime
# from copy import deepcopy


# simple scheduler
# 1. Draw random plan from population, controlling for selected demographic category
# 2. Select location

class Scheduler:
    """
    Build daily plan
    """
    def __init__(self, person, planner, control_fields = None):

        attributes = person.attributes
        if control_fields is not None:
            attributes = {key:value for key, value in attributes.items() if key in control_fields}
            
        self.plan = planner.draw_PAM_plan(attributes)
        self.person = person

        person.plan = self.plan


        print('Person', person, '|Hzone',person.attributes['hzone'],'| Age:', person.attributes['age'], ', Gender:', person.attributes['gender'])

        print('Selected plan:', self.plan)
        print('-'*50)

        # select locations
        # assign home location
        for act in person.activities:
            if act.act == 'home':
                act.location = Location(area = person.attributes['hzone'])
            else:
                # select location
                act.location = Location(area = 'NA')


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
        ## self.select_activity()
        # self.select_start_time()
        # self.select_duration()
        # self.select_location()
        # self.select_mode()

        pass