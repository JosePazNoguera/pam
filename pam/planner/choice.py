import pandas as pd
import numpy as np

def logit_choice(utilities):
    """
    Simple logit choice model
    :params pd.Series utilities: a Pandas series, whose index is the choice labels, and values the utilities
    """
    weights = np.exp(utilities) / np.exp(utilities).sum()
    choice = utilities.sample(weights = utilities).index[0]
    return choice

def logsum(utilities):
    """
    Logsum calculation (log of the sum of exponentiated utilities)
    :params pd.Series utilities: a Pandas series, whose index is the choice labels, and values the utilities
    """
    return np.log(np.exp(utilities).sum())


def sample_weighted(weights, n=1):
    """
    Draw a weighted sample from a Pandas series    
    """
    sample = weights.sample(n, weights=weights).index.to_list()
    if n == 1:
        return sample[0]
    return sample

    
def p_location(attractions, impedance, l):
    """
    Select a workplace destination
    :params pd.Series attractions: destination scale measure (ie jobs)
    :params pd.Series impedance: impedance measure (ie distance to zone)
    :params float l: a destination choice distance parameter
    
    :returns: probabilities of workplace
    """
    p = attractions * np.exp(l*impedance) / np.exp(l*impedance).sum()
    p = p / p.sum() # normalise
    return p


def p_expected_impedance(attractions, impedance, expected_impedance, k=-0.5, x0=10):
    """
    """
    p = attractions * p_logistic(np.abs(impedance-expected_impedance), k, x0)
    p = p / p.sum() # normalise
    return p

def p_logistic(x, k, x0):
    """
    Logistic function probability. p(0) = 1, p(large)->0, p(x0)=0.5
    :params float k: Curve steepness parameter
    :params float x0: curve midpoint

    :returns: probability (0-1)
    """
    p = 1 / (1+np.exp(-k*(x-x0)))
    return p