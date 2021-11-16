import pandas as pd
from pathlib import Path
from numpy import sqrt

def read_participant_csv(file):
    return (pd.read_csv(file, header=None, names = ['time_step', 'x', 'y', 'z'])
            .assign(participant = int(Path(file).stem)))


def add_acceleration_magnitude(dataf):
    return (dataf
            .assign(r=lambda d: sqrt(d['x']**2 + d['y']**2 + d['z']**2)))


def add_time_group(dataf, time_window = 5):
    return (dataf
            .assign(time_group = lambda d: d.time_step.astype(int) // time_window))


def calculate_attributes(dataf):
    return (dataf
            .groupby(['participant', 'time_group'], as_index=False)
            .agg(x_mean=('x', 'mean'), 
                 x_median=('x', 'median'),
                 x_sd=('x', 'std'),
                 y_mean=('y', 'mean'), 
                 y_median=('y', 'median'),
                 y_sd=('y', 'std'),
                 z_mean=('z', 'mean'), 
                 z_median=('z', 'median'),
                 z_sd=('z', 'std'),
                 r_mean=('r', 'mean'), 
                 r_median=('r', 'median'),
                 r_sd=('r', 'std'),
                 time_mean_diff = ('time_step', lambda x: x.diff().mean()))
            .drop(['time_group'], axis=1))   