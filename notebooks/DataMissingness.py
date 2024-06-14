from collections import deque

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression


class MissingnessDetector:
    """Evaluates the score for type of missing data. Type of missing data : MCAR, MAR, MNAR"""

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.data_window = deque(maxlen=window_size)
        self.missing_labels_window = deque(maxlen=window_size)
        self.mcar_scores_random_test = []
        self.mcar_scores_chi2_test = []
        self.mar_scores = []
        self.mnar_scores = []

    def update(self, new_data):
        self.data_window.append(new_data)
        missing_labels = {col: int(pd.isnull(new_data[col])) for col in new_data}
        self.missing_labels_window.append(missing_labels)

        if len(self.data_window) == self.window_size:
            self.evaluate_missingness()

    def evaluate_missingness(self):
        """Get score for each type of missing data : MCAR, MAR, MNAR"""
        df_window = pd.DataFrame(self.data_window)
        missing_labels_df = pd.DataFrame(self.missing_labels_window)

        # MCAR Test
        self.evaluate_MCAR_with_random_test(df_window)
        self.evaluate_MCAR_with_chi2_test(df_window)

        # MAR Test
        for col in df_window.columns:
            if missing_labels_df[col].sum() > 0:
                self.evaluate_MAR(df_window, col, missing_labels_df[col])

        # MNAR Test
        self.evaluate_MNAR()

    def evaluate_MCAR_with_random_test(self, df):
        """
        MCAR (Missing Completely at Random): The missing data are independent of both observed and unobserved data.
        In other words, the missingness does not depend on any values in the dataset.

        Simple random test: are the missing values distributed randomly?
        """
        missing_counts = df.isnull().sum()
        total_counts = len(df)
        missing_ratios = missing_counts / total_counts
        self.mcar_scores_random_test.append(missing_ratios.mean())

    def evaluate_MCAR_with_chi2_test(self, df):
        """
        MCAR (Missing Completely at Random): The missing data are independent of both observed and unobserved data.
        In other words, the missingness does not depend on any values in the dataset.

        Little’s MCAR Test: Implemented using the chi-squared test (chi2_contingency).
        If the p-value is high, we fail to reject the null hypothesis that the data is MCAR.
        High p-values suggest that the data is MCAR.
        """
        complete_cases = df.dropna()
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                observed_freq = pd.crosstab(
                    df[col].isnull(), complete_cases.notnull().sum(axis=1)
                )
                chi2, p, _, _ = chi2_contingency(observed_freq)
                self.mcar_scores_chi2_test.append(p)

    def evaluate_MAR(self, df, target_column, missing_labels):
        """
        MAR (Missing at Random): The missing data are related to some of the observed data
        but not the missing data itself.
        The probability of missingness depends on observed data but not on unobserved data.

        Fit a logistic regression model where the dependent variable indicates whether a value is missing,
        and the independent variables are other observed values in the dataset.
        A significant relationship between the independent variables and the dependent variable indicates MAR.
        Higher scores indicate that missingness is more predictable from observed data, suggesting MAR.
        """
        df = df.copy()
        df["missing"] = missing_labels
        df = df.drop(columns=[target_column])
        y = df["missing"]
        X = df.drop(columns=["missing"]).fillna(-9999)

        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(X, y)

        score = log_reg.score(X, y)
        self.mar_scores.append(score)

    def evaluate_MNAR(self):
        """
        MNAR (Missing Not at Random): The missing data are related to the unobserved data.
        The probability of missingness is related to the values that are missing.

        MNAR Test potential approaches:
            1. Heuristic Methods: Often based on domain knowledge and contextual understanding of the data.
            2. MNAR Indicators: If neither MCAR nor MAR hold, data might be MNAR.

        Implemented point 2 above with help of maximum of MCAR and MAR
        MNAR score = 1 - max(MCAR and MAR)
        """
        if not self.mcar_scores_random_test or not self.mar_scores:
            raise ValueError(
                "MCAR and MAR scores must be calculated before MNAR score."
            )

        last_mcar_score = self.mcar_scores_random_test[-1]
        last_mar_score = self.mar_scores[-1]

        max_val_mcar_mar = max(last_mcar_score, last_mar_score)
        mnar_score = 1 - max_val_mcar_mar
        self.mnar_scores.append(mnar_score)

    def get_results(self):
        return {
            "MCAR (Random Test)": self.mcar_scores_random_test,
            "MCAR (Chi2 Test)": self.mcar_scores_chi2_test,
            "MAR": self.mar_scores,
            "MNAR": self.mnar_scores,
        }


if __name__ == "__main__":
    # Example usage
    detector = MissingnessDetector(window_size=100)

    # Simulating a data stream
    for i in range(1000):
        new_data = {
            "A": np.random.choice([1, 2, 3, np.nan]),
            "B": np.random.choice([1, 2, 3, 4, 5, np.nan]),
            "C": np.random.choice([1, 2, 3, 4, 5]),
        }
        detector.update(new_data)

    results = detector.get_results()
    print("MCAR Scores (Random Test): ", results["MCAR (Random Test)"])
    print("MCAR Scores (Chi2 Test): ", results["MCAR (Chi2 Test)"])
    print("MAR Scores: ", results["MAR"])
    print("MNAR Scores: ", results["MNAR"])
