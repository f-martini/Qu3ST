import matplotlib.pyplot as plt
import numpy as np


def get_line(df, pivot, values):
    grouped_stats = df.groupby(pivot)[values].agg(['mean', 'std'])
    x = grouped_stats.index.to_series()
    y = grouped_stats['mean']
    sd = grouped_stats['std']
    return x, y, sd


def plot_line(x, y, sd, label, marker='o', linestyle='-',
              color=None):
    plt.plot(x, y, marker=marker, linestyle=linestyle, label=label,
             color=color)
    se = sd / np.sqrt(len(sd))
    plt.fill_between(x, y - se, y + se, alpha=0.2, color=color)
