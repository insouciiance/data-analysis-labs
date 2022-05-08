import math
import matplotlib.pyplot as plt
import scipy.stats as stats
from pandas import DataFrame

def show_pie_diagram(dataset : DataFrame, labels : list, column : str):
    title, diagrams = plt.subplots(figsize=(8, 6))
    diagrams.pie(dataset[column],
                 labels=labels)
    title.suptitle(f"Pie for {column}:", fontsize=20)

    plt.show()

def log_median_significance(dataset : DataFrame, column : str, alpha : float = 0.05):
    N = len(dataset[column])
    df = N - 1
    deviation = dataset[column].std()
    median = dataset[column].median()
    mean = dataset[column].mean()

    # apply Student's t-test
    t = abs(mean - median) / (deviation / math.sqrt(N))
    p_value = stats.t.sf(t, df)
    success = p_value > alpha / 2
    success_message = \
        f"{column}: Expected value and median are the same." if success else \
        f"{column}: Expected value and median are not the same."
    print(success_message)