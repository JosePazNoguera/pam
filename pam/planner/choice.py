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
    