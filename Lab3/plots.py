from pandas import DataFrame
import matplotlib.pyplot as plot

def print_box_plots(dataset : DataFrame, columns : list):
    fig, axs = plot.subplots(1, len(columns), figsize=(16, 8))

    fig.suptitle('Box plots')

    for i, column in enumerate(columns):
        print(dataset[column])
        axs[i].set_title(column)
        axs[i].boxplot(dataset[column])

    plot.show()

def print_histograms(dataset : DataFrame, columns : list):
    fig, axs = plot.subplots(1, len(columns), figsize=(16, 8))

    fig.suptitle('Histograms')

    for i, column in enumerate(columns):
        print(dataset[column])
        axs[i].set_title(column)
        axs[i].hist(dataset[column])

    plot.show()