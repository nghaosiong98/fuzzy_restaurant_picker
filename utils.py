import matplotlib.pyplot as plt


def plot_mf(universe, low=None, medium=None, high=None, xlabel='', ylabel=''):
    if low is not None:
        plt.plot(universe, low)
    if medium is not None:
        plt.plot(universe, medium)
    if high is not None:
        plt.plot(universe, high)
    plt.xlabel = xlabel
    plt.ylabel = ylabel
    plt.show()
