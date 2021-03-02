
from pam import write
from pam.core import Person

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


class Planner:

    def __init__(self):
        self.population = None
        self.trips = None
        self.activities = None
        self.freq_departure_purp = None
        self.average_od_duration = None
        self.average_od_duration_mode = None
        self.mode_split = None
        self.transition_matrix = None

    def train(self, population):
        """
        Train a choice model from a PAM population.
        This step tries to capture the population’s behaviour patterns. 
        For example, derive probability distributions of activity choice and start time. 

        :params PAM.core.Population population: A PAM population object

        :returns: None
        """
        self.population = population
        self.trips = write.write_benchmarks(population)
        self.activities = write.write_activities(population)
        self.activities['freq'] = 1
        # observed frequncy distributions
        self.freq_departure_purp = write.write_benchmarks(population, dimensions = ['departure_hour','purp'], data_fields=['freq'], aggfunc=sum)

        # weighted average OD duration
        self.average_od_duration = write.write_benchmarks(population, dimensions = ['ozone','dzone'], data_fields=['freq','personhrs'], aggfunc=sum)
        self.average_od_duration['duration'] = self.average_od_duration['personhrs'] / self.average_od_duration['freq']

        # weighted average OD duration by mode
        self.average_od_duration_mode = write.write_benchmarks(population, dimensions = ['ozone','dzone','mode'],  data_fields=['freq','personhrs'], aggfunc=sum)
        self.average_od_duration_mode['duration'] = self.average_od_duration_mode['personhrs'] / self.average_od_duration_mode['freq']

        # mode split
        self.mode_split = write.write_benchmarks(population, dimensions = ['mode'], data_fields=['freq'], aggfunc=sum)

        self.transition_matrix = self.get_transition_matrix(self.trips)


    def get_transition_matrix(self, trips):
        """
        Transition matrix (between activities, by departure hour)
        """

        total_persons = trips.groupby('pid').freq.mean().sum() # weighted number of persons in population
        transition_matrix = trips.copy()
        transition_matrix['purp_previous'] = transition_matrix.groupby('pid').purp.shift(1).fillna('home')
        transition_matrix = transition_matrix.groupby(['arrival_hour','purp','purp_previous']).freq.sum()

        # calculate percentage of people changing activity
        transition_matrix = transition_matrix / total_persons
        transition_matrix = transition_matrix.unstack(level='purp').fillna(0)
        transition_matrix['no_change'] = 1 - transition_matrix.sum(axis=1) # the remaining agents continue their ongoing activity

        # expand matrix to include all hours and purposes
        transition_matrix = self.expand_transition_matrix(transition_matrix)
         
        return transition_matrix



    def create_dimensions(self, *args):
        """
        Create all possible combinations
        :params lists *args: Arbitrary number of lists containing the various dimensions
        """
        dimensions=[]
        dim_generator = itertools.product(*args)
        for dim in dim_generator :
            dimensions.append([x for x in dim])
        dimensions = pd.DataFrame(dimensions)

        return dimensions

    def expand_transition_matrix(self, df):
        """
        Expand transition matrix to include all hours and purposes
        """
        purpose_list = set(df.index.get_level_values('purp_previous').unique()).union(df.columns)
        hour_list = [x for x in range(0,24)]
        dimensions = self.create_dimensions(hour_list, purpose_list)
        dimensions.columns = ['arrival_hour','purp_previous']
        dimensions = dimensions.set_index(['arrival_hour','purp_previous'])
        expanded_matrix = pd.concat([dimensions, df], axis=1)

        # for missing transitions in the diary, assume no activity change
        expanded_matrix['no_change'] = expanded_matrix['no_change'].fillna(1)
        expanded_matrix = expanded_matrix.fillna(0)

        return expanded_matrix



    def generate_person(self, weighted_on):
        """
        """
        p = self.activities.groupby(weighted_on).freq.sum()
        p = p / p.sum()
        p = p.sample(weights = p.values)
        attributes = p.reset_index().drop(columns=['freq']).to_dict('records')[0]
        person = Person('a', attributes=attributes)
        return person


    ##### plots ################################################################################

    def plot_transition_matrix(self, hour):
        """
        Heatmap of the transition matrix for a specified hour
        """
        plt.figure(figsize=(10,5))
        sns.heatmap(self.transition_matrix.loc[hour], annot=True, fmt='.1%', cmap='Blues', linewidths=.5)
        plt.title('Transition matrix, {:.0f}:00-{:.0f}:00'.format(hour, hour+1))
        plt.ylabel('From activity')
        plt.xlabel('To activity')

    def plot_activity_duration_summary(self):
        """
        Histogram of activity durations by purpose
        """
        for act in self.activities.act.unique():
            sns.histplot(self.activities[self.activities.act==act].duration_minutes, kde=True)
            plt.grid()
            plt.title('Activity Duration, {}'.format(act))
            plt.xlabel('Duration (minutes)')
            plt.show()
