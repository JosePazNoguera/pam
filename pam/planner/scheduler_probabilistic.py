
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
        min_duration = 5/60 # minumum activity duration of 5 minutes 
        min_duration_home =30/60 # 30 min minumum stay at home 
        self.time = 0

        start_time = 0
        
        # create plans 
        plan = planner.draw_plan([person.attributes['age'],person.attributes['gender']]).split('-->')

        print('Person', person, '| Age:', person.attributes['age'], ', Gender:', person.attributes['gender'])

        print('Selected plan:',plan)
        print('-'*50)

        for idx, act in enumerate(plan):
            if self.time < 24:
                # TODO: how to deal with overflowing plans? Sample controlling for start time?
                if idx==0:
                    print('Activity: home, 0')
                    # start_time = 0
                else:
                    if act!='home':
                        print('-'*50)
                        print('Activity:', act)
                        # sample start time
                        # make sure we are only sampling forwards in time
                        while start_time < self.time+0.001: 
                            start_time = planner.draw_start_time(act)

                        duration = -1
                        while duration < min_duration:
                            duration = planner.draw_duration(act)
                        
                        self.time = start_time + duration

                        print('start time: {:.2f}'.format(start_time))
                        print('duration: {:,.2f} hours'.format(duration))
                        print('end time: {:.2f}'.format(self.time))
                        print('-'*50)


                        start_time_ts = minutes_to_datetime(int(start_time*60))
                        end_time_ts = minutes_to_datetime(int(self.time*60))
                        person.add(Activity(seq=idx, act=act, area='a',
                            start_time=start_time_ts,
                            end_time=end_time_ts
                            )
                        )
                        person.add(Leg(seq=idx,mode='car',start_area='a',end_area='b', start_time=end_time_ts, end_time=end_time_ts))

                    else:
                        print('Activity: home')                   
                        # print('start time: {:.2f}'.format(self.time))
                        self.time += min_duration_home
                        # print('end time: {:.2f}'.format(self.time))



        # sequencer = Sequencer(person, planner)
        # self.schedule = [self.record_values(sequencer)]
        # self.person = person
        # for step in sequencer:
        #     self.schedule.append(self.record_values(step))

        # # add PAM plans
        # self.add_plans()

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
        ## self.select_activity()
        # self.select_start_time()
        # self.select_duration()
        # self.select_location()
        # self.select_mode()

        pass