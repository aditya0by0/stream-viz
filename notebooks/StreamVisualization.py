import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from river.drift import ADWIN, DriftRetrainingClassifier

# from river import DriftDetector
from river.linear_model import LogisticRegression
from river.metrics import Accuracy
from sklearn.preprocessing import OneHotEncoder


class StreamVisualization:

    def __init__(self, df: pd.DataFrame, y: pd.DataFrame):
        self.df = df
        self.y = y

    def plot(self, timepoint_start: int = 0, timepoint_end: int = 10):
        # Create main plot with subplots
        fig, axes = plt.subplots(
            nrows=len(self.df.columns), ncols=1, figsize=(8, 6), sharex=True
        )

        df = self.df.iloc[timepoint_start:timepoint_end]

        # Plot each feature against its index
        for i, col in enumerate(df.columns):
            axes[i].plot(df.index, df[col], marker="o", linestyle="-")
            axes[i].set_ylabel(col)
            axes[i].grid(True)

        # Set x-axis label for the last subplot
        axes[-1].set_xlabel("TimePoint")

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plot
        plt.show()

    def detect_concept_drift(self):
        # Instantiate a logistic regression classifier
        classifier = LogisticRegression()

        drift_detector = ADWIN()

        # Instantiate a drift retraining classifier with the logistic regression classifier and drift detector
        retraining_classifier = DriftRetrainingClassifier(
            model=classifier,
            drift_detector=drift_detector,
        )

        # Create a data stream (replace with your own data stream)
        stream = zip(self.df, self.y)

        # Metrics for monitoring performance
        accuracy = Accuracy()

        # Train and evaluate the drift retraining classifier on the data stream
        for x, y in stream:
            # Predict the class label and update the model
            x_dict = x.to_dict(orient="records")[0]  # Assuming x contains a single row
            y_pred = retraining_classifier.predict_one(x_dict)
            retraining_classifier.learn_one(x, y)

            # Update the drift detector and check for drift
            drift_detected = retraining_classifier.drift_detector_["adwin"].add_element(
                int(y != y_pred)
            )

            # Update performance metrics
            accuracy.update(y, y_pred)

            # If drift is detected, retrain the model
            if drift_detected:
                retraining_classifier.reset()

            # Print performance metrics periodically (e.g., every 100 samples)
            if stream.n_samples % 100 == 0:
                print(f"Accuracy: {accuracy.get():.4f}")


def __get_data():
    df_cfpdss = pd.read_csv(
        "C:/Users/HP/Desktop/github-aditya0by0/stream-viz/data/cfpdss.csv"
    )
    X_df = df_cfpdss.drop(columns="class")
    y_df = df_cfpdss[["class"]]
    categorical_columns = X_df.select_dtypes(include=["object"]).columns.tolist()
    X_df_categorical = X_df[categorical_columns]
    encoder = OneHotEncoder(sparse_output=False, drop="if_binary", dtype=np.int32)
    one_hot_encoded = encoder.fit_transform(X_df_categorical)
    columns = encoder.get_feature_names_out()
    X_df_cat_one_hot = pd.DataFrame(one_hot_encoded, columns=columns)
    y_encoder = OneHotEncoder(sparse_output=False, drop="if_binary", dtype=np.int32)
    y_encoded = y_encoder.fit_transform(y_df)
    y_df_one_hot = pd.DataFrame(y_encoded, columns=y_encoder.get_feature_names_out())
    X_df_encoded = pd.concat(
        [
            X_df_cat_one_hot,
            X_df[[col for col in X_df.columns if col not in categorical_columns]],
        ],
        axis=1,
    )
    return X_df_encoded, y_df_one_hot


if __name__ == "__main__":

    X_df, y_df = __get_data()

    stream_ = StreamVisualization(X_df, y_df)
    stream_.detect_concept_drift()
