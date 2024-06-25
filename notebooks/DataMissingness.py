from collections import deque

import numpy as np
import pandas as pd
from pyampute.exploration.mcar_statistical_tests import MCARTest
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression


class MissingnessDetector:
    """
    Evaluates the score for each type of missing data.
    Type of missing data : 1. MCAR, 2. MAR, 3. MNAR

    1. MCAR (Missing Completely at Random):
        The missing data are independent of both observed and unobserved data.
        In other words, the missingness does not depend on any values in the dataset.

        MCAR score can be 0, indicating that missing data are completely at random and there
        is no systematic pattern or dependency on any observed or unobserved data.

        Theoretical maximum value is 1, which would suggest that all missingness can be
        predicted perfectly from observed or unobserved data.

    2. MAR (Missing at Random):
        The missing data are related to some of the observed data but not the missing data itself.
        The probability of missingness depends on observed data but not on unobserved data.

        Theoretical minimum value for MAR score can be 0, indicating that there is no relationship between
        the missingness of data and any observed data. This would imply that missingness is completely random or
        determined by unobserved factors not captured in the model.

        Theoretical maximum value is 1, which would imply a perfect relationship where missingness
        can be completely explained by observed data.

    3. MNAR (Missing Not at Random):
        The missing data are related to the unobserved data.
        The probability of missingness is related to the values that are missing.

        Theoretical minimum value of MNAR score is 0, indicating that missingness is completely at random or is
        related solely to observed or unobserved factors that are not captured in the model.

        Theoretical maximum value is 1, which would indicate that the missingness is entirely due to unobserved
        factors (MNAR), meaning that the missing data are not random and depend solely on the unobserved data.
    """

    def __init__(self, window_size=100, MAR_log_max_iter=100):
        self.window_size = window_size
        # Max iteration parameter for logistic regression in MAR test
        self.MAR_log_max_iter = MAR_log_max_iter
        self.data_window = deque(maxlen=window_size)
        self.missing_labels_window = deque(maxlen=window_size)
        self.mcar_scores_random_test_df = (
            pd.DataFrame()
        )  # DataFrame to store MCAR scores
        self.mcar_scores_chi2_test = pd.DataFrame(columns=["p_value"])
        self.mar_scores_df = pd.DataFrame()  # DataFrame to store MAR scores
        self.mcar_little_test_cls = MCARTest(method="little")

    def update(self, new_data, idx: int):
        self.data_window.append(new_data)
        missing_labels = {col: int(pd.isnull(new_data[col])) for col in new_data}
        self.missing_labels_window.append(missing_labels)

        # if self.mcar_scores_random_test_df.empty:
        #     df_window = pd.DataFrame(self.data_window)
        #     self.mcar_scores_random_test_df = pd.DataFrame(columns=df_window.columns)
        #
        # if self.mcar_scores_chi2_test.empty:
        #     df_window = pd.DataFrame(self.data_window)
        #     self.mcar_scores_chi2_test = pd.DataFrame(columns=df_window.columns)

        if self.mar_scores_df.empty:
            df_window = pd.DataFrame(self.data_window)
            self.mar_scores_df = pd.DataFrame(columns=df_window.columns)

        if len(self.data_window) == self.window_size:
            self.evaluate_missingness(idx)

    def evaluate_missingness(self, idx: int):
        """Get score for each type of missing data : MCAR, MAR, MNAR"""
        df_window = pd.DataFrame(self.data_window)
        missing_labels_df = pd.DataFrame(self.missing_labels_window)

        # MCAR Test
        # self.evaluate_MCAR_with_random_test(df_window, idx)
        self.evaluate_MCAR_with_chi2_test(df_window, idx)

        # # MAR Test
        # for col in df_window.columns:
        #     if missing_labels_df[col].sum() > 0:
        #         self.evaluate_MAR(df_window, col, missing_labels_df[col], idx)

    def evaluate_MCAR_with_random_test(self, df, idx: int):
        """
        Simple random test: are the missing values distributed randomly?
        """
        missing_counts = df.isnull().sum()
        total_counts = len(df)
        missing_ratios = missing_counts / total_counts
        self.mcar_scores_random_test_df.loc[idx] = missing_ratios
        self.mcar_scores_random_test_df.loc[idx, "mean"] = missing_ratios.mean()

    def evaluate_MCAR_with_chi2_test(self, df, idx: int):
        """
        Implementation of Little's MCAR test

        Parameters
        ----------
        X : Matrix of shape `(n, m)`
            Dataset with missing values. `n` rows (samples) and `m` columns (features).

        Returns
        -------
        pvalue : float
            The p-value of a chi-square hypothesis test.
            Null hypothesis: data is Missing Completely At Random (MCAR).
            Alternative hypothesis: data is not MCAR.
            If the p-value is lower than the significance level (typically 0.05), we reject the null hypothesis,
            concluding that the data is not MCAR.
            If the p-value is higher than the significance level, we fail to reject the null hypothesis,
            suggesting that the data is MCAR.
        """
        # increase step size, with min 25% overlap
        p_value = self.mcar_little_test_cls.little_mcar_test(df)
        self.mcar_scores_chi2_test.loc[idx, "p_value"] = p_value

    def evaluate_MAR(self, df, target_column, missing_labels, idx):
        """
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

        log_reg = LogisticRegression(max_iter=self.MAR_log_max_iter)
        log_reg.fit(X, y)

        score = log_reg.score(X, y)
        self.mar_scores_df.loc[idx, target_column] = score

    def evaluate_MNAR(self):
        """
        MNAR Test potential approaches:
            1. Heuristic Methods: Often based on domain knowledge and contextual understanding of the data.
            2. MNAR Indicators: If neither MCAR nor MAR hold, data might be MNAR.

        Implemented point 2 above with help of maximum of MCAR and MAR
        MNAR score = 1 - max(MCAR and MAR)
        """
        if self.mcar_scores_chi2_test.empty or self.mar_scores_df.empty:
            raise ValueError(
                "MCAR and MAR scores must be calculated before MNAR score."
            )

        # Calculate MNAR scores using vectorized operations
        max_val_mcar_mar = np.maximum(
            self.mcar_scores_random_test_df, self.mar_scores_df
        )
        mnar_scores = 1 - max_val_mcar_mar

        self.mnar_scores_df = pd.DataFrame(
            data=mnar_scores, index=self.mar_scores_df.index
        )

    def get_results(self):
        # Add mean of each row as a new column
        self.mar_scores_df["mean"] = self.mar_scores_df.mean(axis=1)
        # self.evaluate_MNAR()

        return {
            # "MCAR (Random Test)": self.mac,
            "MCAR (Chi2 Test)": self.mcar_scores_chi2_test,
            # "MAR": self.mar_scores_df,
            # "MNAR": self.mnar_scores_df,
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
        detector.update(new_data, idx=i)

    results = detector.get_results()
    print("MCAR Scores (Random Test): ", results["MCAR (Random Test)"])
    print("MCAR Scores (Chi2 Test): ", results["MCAR (Chi2 Test)"])
    print("MAR Scores: ", results["MAR"])
    print("MNAR Scores: ", results["MNAR"])
