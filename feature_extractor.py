import numpy as np
import pandas as pd

# two experiments

# use all 79 dimensions mentioned in the challenge. if that is the case then feauture vector will be a
# use only one sensor

# frame : 2-D numpy array containing raw sensor data.; num_samples * num_channels

ecdf_bins = 15

def calculate_features(frame):
    # for now calculate ecdf and mean of each channel and concatenate them
    data_df = pd.DataFrame(frame,index= None)
    return np.array(calculate_ecdf(data_df, ecdf_bins) + calculate_mean(data_df))

def calculate_ecdf(X, n):
    '''Calculate empirical cumulative distribution function (ECDF) from X at n points
    Ref: 1.Nils Y. Hammerla et. al 2013. On preserving statistical characteristics of accelerometry data using their empirical cumulative distribution.

    Parameters
    ----------
    X : pandas.Dataframe
        Sensor data

    n : integer
        Number of sample points

    Returns
    -------
    ecdf_vec : list
        1-D list of
    '''
    # Ref: http://stackoverflow.com/questions/14006520/ecdf-in-python-without-step-function
    # Ref: http://stackoverflow.com/questions/21887625/python-empirical-distribution-function-ecdf-implementation
    # Ref: http://statsmodels.sourceforge.net/stable/generated/statsmodels.tools.tools.ECDF.html

    import statsmodels.api as sm
    ecdf_vec = []
    sample = np.linspace(0.0, 1.0, n)
    for column in X:
        X_sort = np.sort(X[column])
        # Random noise is added to aid with float comparisons for CDF calculation
        noisy_data = X_sort + np.random.randn(len(X_sort)) * 0.001
        ecdf = sm.distributions.ECDF(noisy_data)
        f = ecdf(noisy_data)
        # Calculate inverse CDF at linearly spaced points to get a sparse representation of data
        li_col = np.interp(sample, f, noisy_data)
        ecdf_vec = ecdf_vec + li_col.tolist()

    return ecdf_vec


def calculate_mean(X):
    '''Compute mean per sensor channel

    Parameters
    ----------
    X : pandas.Dataframe
        Sensor data

    Returns
    -------
    mean_vec : list
        1-D list of mean values per sensor channel
    '''
    mean_vec = []
    for column in X:
        mean_vec.append(np.mean(X[column]))

    return mean_vec
