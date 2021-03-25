
from pam import write
from pam.core import Person

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import scipy
import itertools
from pam import variables
from pam.planner import choice

import numpy as np


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
        self.trip_duration_pdf = None
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
        self.trip_duration_pdf = self.get_trip_duration_pdf()
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
        plan_freqs = self.activities.groupby(control_groups).apply(self.get_plan_frequencies)
        plan_freqs.index.rename(level=[-1], names=['plan'], inplace=True)
        return plan_freqs

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
        Gausian KDE probability function of activity duration (in hours) for each purpose
        """
        pdf_dict = {}
        for act in self.activities.act.unique():
            pdf_dict[act] = scipy.stats.gaussian_kde(self.activities[self.activities.act==act].duration_minutes/60)

        return pdf_dict

    def get_trip_duration_pdf(self):
        """
        Gausian KDE probability function of travel time (in hours) for each purpose
        """
        pdf_dict = {}
        for purpose in self.trips.purp.unique():
            pdf_dict[purpose] = scipy.stats.gaussian_kde(self.trips[self.trips.purp==purpose].duration/60)

        return pdf_dict


    def get_start_time_pdf(self):
        """
        Gausian KDE probability function of each purpose
        """
        pdf_dict = {}
        for act in self.activities.act.unique():
            pdf_dict[act] = scipy.stats.gaussian_kde(self.activities[self.activities.act==act].start_minute / 60)

        return pdf_dict

    ##### tour analysis ################################################################################
    def get_tours(self, plan, target_act='home'):
        """
        Extract tours from an activity sequence
        :params list plan: A sequence (list) of activities (ie ['home', 'escort_education', 'home', 'work', 'home'])
        :params str target_act: The "base" of the tour (ie if target_act=='home', then it will home-based tours are returned)
        """
        tours = []
        tour = None
        for i, act in enumerate(plan):
            if act == target_act:
                if tour is not None:
                    tours.append(tour)
                tour = []
            elif tour is not None:
                tour.append(act)

        return tours

    def get_home_tours(self):
        """
        Split persons' activities into tour lists
        :returns: pd.Series with person ID as index, and a list of tour lists as values
        """
        return self.activities.groupby('pid').act.apply(list).apply(lambda x: self.get_tours(x,'home'))

    ##### sampling ################################################################################
    def draw_kde(self, pdf, n=1):
        """
        Draw a sample from a Kernel Density Function
        """
        return pdf.resample(n)[0][0]

    def draw_duration(self, act):
        """
        Draw a duration sample for a specified activity purpose
        :params str act: activity purpose (ie 'work')
        """
        return self.draw_kde(self.duration_pdf[act])

    def draw_trip_duration(self, act):
        """
        Draw a duration sample for a specified activity purpose
        :params str act: activity purpose (ie 'work')
        """
        return self.draw_kde(self.trip_duration_pdf[act])

    def draw_start_time(self, act):
        """
        Draw a duration sample for a specified activity purpose
        :params str act: activity purpose (ie 'work')
        """
        return self.draw_kde(self.start_time_pdf[act])

    def draw_plan(self, group):
        """
        Draw a plan, sampling for the specified demographic category
        :params list group: A list of the subground to sample for. For example if self.plan_frequencies_group is grouped by age and gender, it can be ['female','60plus']
        """
        return choice.sample_weighted(self.plan_frequencies_group.loc[tuple(group)])

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


    def plot_kde(self, pdf, start=0, end=24, title='Probability density', xlabel='Time', ylabel='Probability', figsize=(10,6)):
        """
        Plot kernel density function probabilities
        :params scipy.stats.gaussian_kde pdf: s KDE density function
        :params int start: start hour 
        :params int end: end hour 
        """
        x = np.linspace(start,end,10*end)
        plt.figure(figsize=figsize)
        plt.plot(x, pdf(x))#, marker='.')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(start, end)
        plt.grid()
        # plt.show()

    def plot_duration_kde(self):
        """
        Plot all duration density functions
        """
        for act in self.duration_pdf:
            self.plot_kde(self.duration_pdf[act], title = 'Duration probability, {}'.format(act), figsize=(8,3))

    def plot_trip_duration_kde(self):
        """
        Plot all duration density functions
        """
        for act in self.trip_duration_pdf:
            self.plot_kde(self.trip_duration_pdf[act], title = 'Trip duration probability, {}'.format(act), figsize=(8,3), end=2, xlabel='Time (hours')

    def plot_start_time_kde(self):
        """
        Plot all duration density functions
        """
        for act in self.start_time_pdf:
            self.plot_kde(self.start_time_pdf[act], title = 'Start time probability, {}'.format(act), figsize=(8,3))

    def plot_plan_frequencies(self, act_seqs, print_results, n, title, figsize=(10,7)):
        """
        Plot top plan frequencies
        :params pd.Series act_seqs: Activity sequence frequencies
        """
        # act_seqs = self.get_plan_frequencies(self.activities)# * 100
        if print_results:
            print('Most common activity sequences')
            print('-'*100)
            print(act_seqs[:n])
            print('-'*100)
        fig, ax = plt.subplots(1,1, figsize=figsize)
        act_seqs[:n].sort_values(ascending=True).plot(kind='barh', ax=ax)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
        plt.title(title)
        plt.grid()
        plt.xlabel('% of plans')
        # plt.show()

    def plot_plan_frequencies_group(self, groupby=None, print_results=False, n=10, title='Most common activity sequences'):
        """
        Plot top n plan frequencies
        """
        if groupby is None:
            act_seqs = self.get_plan_frequencies(self.activities)
            self.plot_plan_frequencies(act_seqs=act_seqs, print_results=print_results, n=n, title=title)
        else:
            act_seqs = self.get_plan_frequencies_group(groupby)
            for i, igroup in act_seqs.groupby(level=act_seqs.index.names[:-1]):
                label = i if isinstance(i, str) else ', '.join(i)
                self.plot_plan_frequencies(act_seqs=igroup.droplevel(igroup.index.names[:-1]), print_results=print_results, n=n, title=title+', '+label)

    def plot_tour_frequency_home(self):
        """
        Plot home-based tours frequency
        """
        home_tours = self.get_home_tours()
        ((pd.Series([x for y in home_tours for x in y]).\
            apply(lambda x: '-'.join(x)).value_counts())/len(home_tours)).\
            sort_values(ascending=False).head(20).sort_values(ascending=True).\
            plot(kind='barh', figsize=(12,17))

        plt.title('Home-based tour frequency')
        plt.xlabel('% of agents undertaking the tour in the day')
        plt.grid()
        plt.show()

    def plot_trip_duration_cdf(self, control='purp'):
        """
        Plot the cumulative frequency of trip durations 
        :params str control: a field to group by
        """
        trip_durations = self.trips.groupby([control,'duration']).freq.sum()
        trip_durations = trip_durations.groupby(level=control).cumsum()
        trip_durations = trip_durations / trip_durations.groupby(level=[control]).max()
        trip_durations = trip_durations.reset_index().rename(columns={'freq':'trips'})

        plt.figure(figsize=(17,10))
        for c in trip_durations[control].unique():
            trip_durations[trip_durations[control]==c].set_index(['duration']).\
                trips.plot(kind='line', label=c, alpha = 0.8)

        plt.title('Cumulative frequency of trip duration by {}'.format(control))
        plt.ylabel('% of trips')
        plt.xlabel('Duration (minutes)')
        plt.xlim(0,120)
        plt.ylim(0,1)
        plt.grid()
        plt.legend()
        plt.show()

