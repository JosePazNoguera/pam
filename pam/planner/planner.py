
from pam import write
from pam.core import Person

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import itertools
from pam import variables


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
        self.duration_pdf = None
        self.start_time_pdf = None
        self.plan_frequencies = None
        self.plan_frequencies_group = None

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
        self.activities['start_minute'] = (self.activities.start_time - pd.Timestamp(1900,1,1))/pd.Timedelta(minutes=1)

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

        # transition probabilities matrix
        self.transition_matrix = self.get_transition_matrix(self.activities)

        # duration and start time probability density function
        self.duration_pdf = self.get_duration_pdf()
        self.start_time_pdf = self.get_start_time_pdf()

        # activity generation stats
        self.plan_frequencies = self.get_plan_frequencies(self.activities)
        self.plan_frequencies_group = self.get_plan_frequencies_group()


    ##### activity frequencies ################################################################################
    def get_plan_frequencies(self, df):
        """
        Return the frequency of the various plan patterns
        :params pd.DataFrame df: activities table. Has person id ('pid') and sequence ('seq') columns

        :returns: pandas dataframes of plan changes (ie home-->work-->home) and their normalised frequencies
        """
        return df.sort_values(['pid','seq']).groupby(['pid']).act.apply(lambda x:'-->'.join(x)).value_counts(normalize=True)

    def get_plan_frequencies_group(self, control_groups = ['age','gender']):
        return self.activities.groupby(control_groups).apply(self.get_plan_frequencies)

    ##### transition matrix ################################################################################

    def get_transition_hour(self, activities, hour):
        t1 = self.filter_ongoing_activities(activities, hour).set_index(['pid','freq']).act
        t2 = self.filter_ongoing_activities(activities, hour+1).set_index(['pid','freq']).act
        
        # remove any duplicates. TODO: improve this
        t1 = t1.groupby(level=['pid','freq']).head(1)
        t2 = t2.groupby(level=['pid','freq']).head(1)
        

        t = pd.concat([t1, t2], axis=1)
        t.columns = ['purp_previous','purp']
        t['arrival_hour'] = hour
        # t['freq'] = 1
        t = t.reset_index()

        t = t.groupby(['arrival_hour','purp_previous','purp'])['freq'].sum().unstack(level='purp').fillna(variables.SMALL_VALUE)
        t = t.div(t.sum(axis=1), axis=0)
        return t

    def get_transition_matrix(self, activities):
        """
        Transition matrix (between activities, by departure hour)
        """

        # # total_persons = trips.groupby('pid').freq.mean().sum() # weighted number of persons in population
        # transition_matrix = trips.copy()
        # transition_matrix['purp_previous'] = transition_matrix.groupby('pid').purp.shift(1).fillna('home')
        # transition_matrix = transition_matrix.groupby(['arrival_hour','purp','purp_previous']).freq.sum()       

        # # calculate percentage of people changing activity
        # # transition_matrix = transition_matrix / total_persons
        # transition_matrix = transition_matrix.unstack(level='purp').fillna(0)
        # totals = self.get_acts_breakdown(self.activities).stack().reset_index()
        # totals.columns = ['arrival_hour','purp_previous','total']
        # totals = totals.set_index(['arrival_hour','purp_previous'])
        # transition_matrix = transition_matrix.join(totals)
        # transition_matrix = transition_matrix.div(transition_matrix.total, axis=0)

        # transition_matrix['no_change'] = 1 - transition_matrix.sum(axis=1) # the remaining agents continue their ongoing activity

        transition_matrix = pd.concat([self.get_transition_hour(activities, hour) for hour in range(23)], axis=0)

        # expand matrix to include all hours and purposes
        transition_matrix = self.expand_transition_matrix(transition_matrix)

        # sort columns 
        transition_matrix = transition_matrix[sorted(transition_matrix.columns)]

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

        :returns: pandas DataFrame
        """
        purpose_list = set(df.index.get_level_values('purp_previous').unique()).union(df.columns)
        hour_list = [x for x in range(0,24)]
        dimensions = self.create_dimensions(hour_list, purpose_list)
        dimensions.columns = ['arrival_hour','purp_previous']
        dimensions = dimensions.set_index(['arrival_hour','purp_previous'])
        expanded_matrix = pd.concat([dimensions, df], axis=1)

        # for missing transitions in the diary, assume no activity change
        # expanded_matrix['no_change'] = expanded_matrix['no_change'].fillna(1)
        expanded_matrix = expanded_matrix.fillna(variables.SMALL_VALUE)

        return expanded_matrix

    def get_duration_pdf(self):
        """
        Gausian KDE probability function of each purpose
        """
        pdf_dict = {}
        for act in self.activities.act.unique():
            pdf_dict[act] = scipy.stats.gaussian_kde(self.activities[self.activities.act==act].duration_minutes/60)

        return pdf_dict


    def get_start_time_pdf(self):
        """
        Gausian KDE probability function of each purpose
        """
        pdf_dict = {}
        for act in self.activities.act.unique():
            pdf_dict[act] = scipy.stats.gaussian_kde(self.activities[self.activities.act==act].start_minute)

        return pdf_dict


    ##### generative functions ################################################################################

    def generate_person(self, weighted_on):
        """
        Randomly create a PAM person (weighted sampling)
        :params list weighted_on: The dimensions to weight on
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
        plt.figure(figsize=(14,12))
        sns.heatmap(self.transition_matrix.loc[hour], annot=True, fmt='.1%', cmap='Blues', linewidths=.5)
        plt.title('Transition matrix, {:.0f}:00-{:.0f}:00'.format(hour, hour+1))
        plt.ylabel('From activity')
        plt.xlabel('To activity')


    def plot_activity_duration_summary(self):
        """
        Histogram of activity durations by purpose
        """
        for act in self.activities.act.unique():
            sns.histplot(self.activities[self.activities.act==act].duration_minutes / 60, kde=True)
            plt.grid()
            plt.title('Activity Duration, {}'.format(act))
            plt.xlabel('Duration (hours)')
            plt.show()

    def plot_activity_start_summary(self):
        """
        Histogram of activity start time by purpose
        """
        for act in self.activities.act.unique():
            sns.histplot(self.activities[self.activities.act==act].start_time.dt.hour, kde=True)
            plt.grid()
            plt.title('Activity start time, {}'.format(act))
            plt.xlabel('Start hour')
            plt.xlim(0, 24)
            plt.show()

    def filter_ongoing_activities(self, df, hour):
        """
        Filter the self.activity table to return ongoing activities at the start of a given hour
        """
        return df[(df.start_time <= pd.Timestamp(1900,1,1,hour))&            
                    (df.end_time > pd.Timestamp(1900,1,1,hour))]

    def get_acts_breakdown(self, df):
        """
        Get the breakdown of activities across the day
        """
        act_breakdown = []
        for hour in range(24):
            # ongoing activities at the start of each hour
            hour_breakdown = self.filter_ongoing_activities(df, hour).groupby('act').freq.sum()
            hour_breakdown['hour'] = hour
            act_breakdown.append(hour_breakdown)
            
        act_breakdown = pd.DataFrame(act_breakdown).fillna(0).set_index('hour')

        return act_breakdown

    def plot_activity_breakdown(self):
        """
        Plot the breakdown of ongoing activities for each hour
        """
        act_breakdown = self.get_acts_breakdown(self.activities)
        act_breakdown = act_breakdown.div(act_breakdown.sum(axis=1), axis=0) # convert to percentage breakdown
        
        # plot
        act_breakdown.sort_index(ascending=False).plot(kind='barh', stacked=True, figsize=(17,17))
        plt.title('Breakdown of activities by hour')
        plt.ylabel('Hour')
        plt.xlabel('% of ongoing activities')
        plt.grid()
        plt.xlim(0,1)
        plt.show()