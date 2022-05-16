import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import check_correlation, get_biggest_clusters, get_kmeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import config
from cleansing import cleanse_data
from helpers import reset_pandas, log_data_frame_info, partition_dataframe, get_stats

reset_pandas(pd)

titanic_data_frame = pd.read_csv(config.DATA_PATH_TITANIC)

titanic_data_frame = titanic_data_frame.drop(columns=["PassengerId", "Name", "Cabin", "Ticket"])

cleanse_data(titanic_data_frame, use_mode = True)

log_data_frame_info(titanic_data_frame)

titanic_data_frame = pd.get_dummies(titanic_data_frame)

titanic_train, titanic_test = train_test_split(titanic_data_frame)

log_data_frame_info(titanic_train)
log_data_frame_info(titanic_test)

x_titanic_train, y_titanic_train = partition_dataframe(titanic_train, "Survived")
x_titanic_test, y_titanic_test = partition_dataframe(titanic_test, "Survived")

decision_tree_classifier = DecisionTreeClassifier(max_depth=5)
decision_tree_mean, decision_tree_score = \
    get_stats(decision_tree_classifier, x_titanic_train, y_titanic_train, x_titanic_test, y_titanic_test)

random_forest_classifier = RandomForestClassifier(max_depth=5)
random_forest_mean, random_forest_score = \
    get_stats(random_forest_classifier, x_titanic_train, y_titanic_train, x_titanic_test, y_titanic_test)

extra_trees_classifier = ExtraTreesClassifier(max_depth=5)
extra_trees_mean, extra_trees_score = \
    get_stats(extra_trees_classifier, x_titanic_train, y_titanic_train, x_titanic_test, y_titanic_test)

ada_boost_classifier = AdaBoostClassifier(learning_rate=0.05)
ada_boost_mean, ada_boost_score = \
    get_stats(ada_boost_classifier, x_titanic_train, y_titanic_train, x_titanic_test, y_titanic_test)

bagging_classifier = BaggingClassifier()
bagging_mean, bagging_score = \
    get_stats(bagging_classifier, x_titanic_train, y_titanic_train, x_titanic_test, y_titanic_test)

gradient_boosting_classifier = GradientBoostingClassifier()
gradient_boosting_mean, gradient_boosting_score = \
    get_stats(gradient_boosting_classifier, x_titanic_train, y_titanic_train, x_titanic_test, y_titanic_test)

print("DecisionTreeClassifier: ", decision_tree_mean, decision_tree_score)
print("RandomForestClassifier: ", random_forest_mean, random_forest_score)
print("ExtraTreesClassifier: ", extra_trees_mean, extra_trees_score)
print("AdaBoostClassifier: ", ada_boost_mean, ada_boost_score)
print("BaggingClassifier: ", bagging_mean, bagging_score)
print("GradientBoostingClassifier: ", gradient_boosting_mean, gradient_boosting_score)

countries_data_frame = pd.read_csv(config.DATA_PATH_COUNTRIES, sep=";", decimal=",", encoding="windows-1251")

log_data_frame_info(countries_data_frame)

cleanse_data(countries_data_frame, negative_columns=["GDP per capita", "Area"])

countries_data_frame["Population density"] = countries_data_frame["Population"] / countries_data_frame["Area"]

clusterization_params = countries_data_frame[["GDP per capita", "Population density"]]

kmeans, elbow = get_kmeans(clusterization_params)

print(get_biggest_clusters(countries_data_frame["Region"], kmeans.predict(clusterization_params)))

fig, axes = plt.subplots(1, 5)

columns = countries_data_frame.select_dtypes(include=np.number)

for idx, column in enumerate(columns):
    axes[idx].set_title(column)
    axes[idx].grid()
    axes[idx].hist(countries_data_frame[column])

plt.show()
