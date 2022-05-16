import numpy as np
from pandas import DataFrame
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score

def reset_pandas(pd):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

def log_data_frame_info(dataset : DataFrame):
    print("Data frame info:")
    dataset.info()

    print("Data format:")
    print(dataset.head(10))
    print(dataset.describe())

def partition_dataframe(data_frame : DataFrame, column : str):
    except_columns = data_frame.loc[:, data_frame.columns != column]
    column = data_frame[column]
    return except_columns, column

def get_stats(model, x_train : DataFrame, y_train : DataFrame, x_test : DataFrame, y_test : DataFrame):
    cross_scores = cross_val_score(model, x_train, y_train)
    cross_scores_mean = cross_scores.mean()
    model.fit(x_train, y_train)
    model_score = model.score(x_test, y_test)
    return cross_scores_mean, model_score

def get_kmeans(data_frame : DataFrame):
    means = []
    max_kernels = 10
    for k in range(1, max_kernels + 1):
        kmeans = KMeans(n_clusters=k, init="random", n_init=10, max_iter=1000)
        kmeans.fit(data_frame)
        means.append(kmeans)
    kl = KneeLocator(range(1, max_kernels + 1), [mean.inertia_ for mean in means], curve="convex", direction="decreasing")
    return means[kl.elbow - 1], kl.elbow

def get_biggest_clusters(df_column, y_kmeans):
    clusters_count = np.unique(y_kmeans).size
    column_counts = [{}] * clusters_count
    for idx, el in enumerate(df_column):
        current_cluster = column_counts[y_kmeans[idx]]
        if el not in current_cluster:
            current_cluster[el] = 0
        current_cluster[el] += 1
    return [(idx, max(counts, key=counts.get)) for idx, counts in enumerate(column_counts)]

def check_correlation(x, y):
    correlation = abs(np.corrcoef(x, y)[0, 1])
    return correlation, correlation > 0.8
