from collections import deque

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class MissingnessDetector:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.data_window = deque(maxlen=window_size)
        self.missing_labels_window = deque(maxlen=window_size)
        self.mcar_scores = []
        self.mar_scores = []

    def update(self, new_data):
        self.data_window.append(new_data)
        missing_labels = {col: int(pd.isnull(new_data[col])) for col in new_data}
        self.missing_labels_window.append(missing_labels)

        if len(self.data_window) == self.window_size:
            self.evaluate_missingness()

    def evaluate_missingness(self):
        df_window = pd.DataFrame(self.data_window)
        missing_labels_df = pd.DataFrame(self.missing_labels_window)

        # MCAR Test
        self.evaluate_mcar(df_window)

        # MAR Test
        for col in df_window.columns:
            if missing_labels_df[col].sum() > 0:
                self.evaluate_mar(df_window, col, missing_labels_df[col])

    def evaluate_mcar(self, df):
        # Simple random test: are the missing values distributed randomly?
        missing_counts = df.isnull().sum()
        total_counts = len(df)
        missing_ratios = missing_counts / total_counts
        self.mcar_scores.append(missing_ratios.mean())

    def evaluate_mar(self, df, target_column, missing_labels):
        df = df.copy()
        df["missing"] = missing_labels
        df = df.drop(columns=[target_column])
        y = df["missing"]
        X = df.drop(columns=["missing"]).fillna(-9999)

        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(X, y)

        score = log_reg.score(X, y)
        self.mar_scores.append(score)

    def get_results(self):
        return {"MCAR": self.mcar_scores, "MAR": self.mar_scores}


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
    print("MCAR Scores: ", results["MCAR"])
    print("MAR Scores: ", results["MAR"])
