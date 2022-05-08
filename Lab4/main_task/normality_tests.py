from enum import Enum
from pandas import DataFrame
import scipy.stats as st

class NormalityTestKind(Enum):
    Shapiro = 0
    Pearson = 1

def get_normality_test_callback(test_kind : NormalityTestKind):
    return \
        st.shapiro if test_kind == NormalityTestKind.Shapiro else \
        st.normaltest

def log_normality_test(
    dataset : DataFrame,
    test_kind : NormalityTestKind,
    alpha : float = 0.05):
    normality_test_callback = get_normality_test_callback(test_kind)
    result, p = normality_test_callback(dataset)
    normality_conclusion = \
        f"'{dataset.name}' is normally distributed." if p > alpha else \
        f"'{dataset.name}' is not normally distributed."
    print(f"Normality test result for '{dataset.name}': ({test_kind.name}): {result}")
    print(normality_conclusion)

def log_normality_tests(
    dataset : DataFrame,
    columns : list,
    test_kind : NormalityTestKind,
    alpha : float = 0.05):
    for column in columns:
        log_normality_test(dataset[column], test_kind, alpha)
