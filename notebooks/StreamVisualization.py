# ------ River imports ---------

# ------ Custom Imports
from collections import deque

import DriftRetrainClassifier
import matplotlib.pyplot as plt
import mplcursors
import mpld3

# ------ Basic python lib imports ----------------
import numpy as np
import pandas as pd
import seaborn as sns
from river import stream

# Drift Detectors
from river.drift import ADWIN, KSWIN, DriftRetrainingClassifier, PageHinkley
from river.drift.binary import *
from river.drift.retrain import DriftRetrainingClassifier
from river.forest import ARFClassifier

# Classifiers
from river.linear_model import LogisticRegression

# Metrics
from river.metrics import F1, Accuracy, BalancedAccuracy, CohenKappa, Precision, Recall
from river.tree import ExtremelyFastDecisionTreeClassifier, HoeffdingTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# sklearn classifiers
from sklearn.neighbors import KNeighborsClassifier

# Others
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# ------ Sk-learn imports -------------


class StreamVisualization:

    def __init__(self, df: pd.DataFrame, y: pd.DataFrame):
        self.df = df
        self.y = y
        self.concept_drifts_timepoints: list = []
        self.feature_drits_timepoints: list = []
        self.prior_drifts_timepoints: list = []

    def adaptive_learning(
        self, model: DriftRetrainClassifier, X_train, y_train, X_test, y_test, metric
    ):
        # Use accuracy as the metric
        i = len(X_train)  # count the number of evaluated data points
        t = []  # record the number of evaluated data points
        m = []  # record the real-time accuracy
        # yt = []  # record all the true labels of the test set
        # yp = []  # record all the predicted labels of the test set

        # Learn the training set
        # for xi1, yi1 in stream.iter_pandas(X_train, y_train):
        #     model.learn_one(xi1, yi1)

        # Predict the test set
        for xi, yi in stream.iter_pandas(X_test, y_test):
            y_pred = model.predict_one(xi)  # Predict the test sample
            model.learn_one(xi, yi, curr_timepoint=i)  # Learn the test sample
            metric.update(yi, y_pred)
            t.append(i)
            m.append(metric.get() * 100)

            i = i + 1

        self.concept_drifts_timepoints = model.drift_timepoints
        return t, m

    def adaptive_learning_2(self, model, X_train, y_train, X_test, y_test, metric):
        # Use accuracy as the metric
        i = len(X_train)  # count the number of evaluated data points
        t = []  # record the number of evaluated data points
        m = []  # record the real-time accuracy
        # yt = []  # record all the true labels of the test set
        # yp = []  # record all the predicted labels of the test set
        my_model = model
        # Learn the training set
        for xi1, yi1 in stream.iter_pandas(X_train, y_train):
            my_model.learn_one(xi1, yi1)

        # Predict the test set
        adwin = ADWIN()
        for xi, yi in stream.iter_pandas(X_test, y_test):
            y_pred = my_model.predict_one(xi)  # Predict the test sample
            my_model.learn_one(xi, yi)  # Learn the test sample
            metric.update(yi, y_pred)
            adwin.update(metric.get())
            t.append(i)
            m.append(metric.get() * 100)

            if adwin.drift_detected:
                # print(f"Change detected at index {i}")
                self.concept_drifts_timepoints.append(i)
                my_model = my_model.clone()
                # adwin._reset()  # Optionally reset the detector

            i = i + 1

        # self.concept_drifts_timepoints = model.drift_timepoints
        return t, m

    def adaptive_learning_3(self, model, X_df, y_df, metric):
        # Use accuracy as the metric
        i = 0  # count the number of evaluated data points
        t = []  # record the number of evaluated data points
        m = []  # record the real-time accuracy
        # yt = []  # record all the true labels of the test set
        # yp = []  # record all the predicted labels of the test set
        my_model = model
        # Learn the training set
        # for xi1, yi1 in stream.iter_pandas(X_train, y_train):
        #     my_model.learn_one(xi1, yi1)

        # Predict the test set
        adwin = ADWIN(delta=0.05)
        # adwin = KSWIN(alpha=0.0001, window_size=200)
        # adwin = PageHinkley()
        for xi, yi in stream.iter_pandas(X_df, y_df):
            y_pred = my_model.predict_one(xi)  # Predict the test sample
            my_model.learn_one(xi, yi)  # Learn the test sample
            metric.update(yi, y_pred)
            adwin.update(metric.get())
            t.append(i)
            m.append(metric.get() * 100)

            if adwin.drift_detected:
                # print(f"Change detected at index {i}")
                self.concept_drifts_timepoints.append(i)
                my_model = my_model.clone()
                # adwin._reset()  # Optionally reset the detector

            i = i + 1

        # self.concept_drifts_timepoints = model.drift_timepoints
        return t, m

    def detect_concept_drift_1(
        model, X_df, y_df, metric, drift_detector, window_size=100
    ):
        timepoint = 0
        metric_score_list = []  # Record the real-time metric
        concept_drifts_timepoints = []
        my_model = model
        my_drift_detector = drift_detector

        metric_window = deque(maxlen=window_size)

        # Stream data, updating the metric and checking for drifts
        for xi, yi in stream.iter_pandas(X_df, y_df):
            y_pred = my_model.predict_one(xi)
            my_model.learn_one(xi, yi)
            metric.update(yi, y_pred)
            curr_metric_val = metric.get() * 100

            # Add the current metric value to the rolling window
            metric_window.append(curr_metric_val)

            # Calculate the average of the metric values in the rolling window
            if len(metric_window) == window_size:
                windowed_metric_val = np.mean(metric_window)
                my_drift_detector.update(windowed_metric_val)
                metric_score_list.append(windowed_metric_val)

                if my_drift_detector.drift_detected:
                    concept_drifts_timepoints.append(timepoint)
                    my_model = my_model.clone()
                    metric_window.clear()  # Empty the window upon drift detection
                    metric = metric.clone()
                    # my_drift_detector._reset()
            else:
                metric_score_list.append(curr_metric_val)

            timepoint += 1

        return concept_drifts_timepoints, metric_score_list

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

    def acc_fig(self, t, m, name):
        plt.rcParams.update({"font.size": 15})
        plt.figure(1, figsize=(10, 6))
        sns.set_style("darkgrid")
        plt.clf()
        plt.plot(t, m, "-b", label="Avg Accuracy: %.2f%%" % (m[-1]))

        plt.legend(loc="best")
        plt.title(name + " on cfpdss dataset", fontsize=15)
        plt.xlabel("Number of samples/Timepoint")
        plt.ylabel("Accuracy (%)")
        # plt.axis([0, 13000, 82, 86])
        for i in range(len(self.concept_drifts_timepoints)):
            # plt.text(self.concept_drifts_timepoints[i] - 500, 100.8, 'Drift ' + str(i), c="red", fontsize=25)
            plt.vlines(
                self.concept_drifts_timepoints[i],
                0,
                100,
                colors="red",
                linewidth=2,
                linestyles="dashed",
            )
        # mplcursors.cursor(hover=True)

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
    y_one_hot = y_encoder.fit_transform(y_df)
    y_encoded = pd.Series(y_one_hot.ravel())
    non_cat_columns = [col for col in X_df.columns if col not in categorical_columns]
    scaler = MinMaxScaler()
    X_non_cat_df = pd.DataFrame(
        scaler.fit_transform(X_df[non_cat_columns]),
        columns=scaler.get_feature_names_out(),
    )
    X_df_encoded = pd.concat(
        [
            X_df_cat_one_hot,
            X_non_cat_df,
        ],
        axis=1,
    )
    return X_df_encoded, y_encoded


if __name__ == "__main__":

    X_df, y_df = __get_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X_df,
        y_df,
        train_size=0.01,
        test_size=0.99,
        shuffle=False,
        random_state=0,
    )
    stream_viz = StreamVisualization(X_df, y_df)
    metrics = [Accuracy, Precision, Recall, F1, BalancedAccuracy, CohenKappa]
    acc = Accuracy()
    acc.clone()
    EFDTC_model = ExtremelyFastDecisionTreeClassifier()
    # model  = EFDTC_model
    arf_model = ARFClassifier()
    # lev_bag_model = LeveragingBaggingClassifier()
    # model = tree.hoeffding_adaptive_tree_classifier.HoeffdingAdaptiveTreeClassifier()
    # model = ADWINBoostingClassifier()
    # model = ADWINBaggingClassifier()
    hoeffding_model = HoeffdingTreeClassifier()

    # model = DriftRetrainClassifier(
    #     model=hoeffding_model,
    #     # drift_detector=drift.binary.DDM()
    #     drift_detector=ADWIN(),
    #     train_in_background=False,
    # )

    # t, m = stream_viz.adaptive_learning(model,
    #                                     X_train,
    #                                     y_train,
    #                                     X_test,
    #                                     y_test,
    #                                     metrics[0]())
    logistic_model = LogisticRegression()

    # t, m = stream_viz.adaptive_learning_2(hoeffding_model,
    #                                     X_train,
    #                                     y_train,
    #                                     X_test,
    #                                     y_test,
    #                                     CohenKappa())

    # t, m = stream_viz.adaptive_learning_3(hoeffding_model, X_df, y_df, CohenKappa())
    # stream_viz.acc_fig(t, m, "sdas")
    # stream_ = StreamVisualization(X_df, y_df)
    # stream_.detect_concept_drift()
# cfpdss concepts:
# n0, n1, n3 ~ N(0, 1)
# n2 = 2 * (n1 + N(0, 1/8))
# n4 = (n1 + n3) / 2 + N(0, 1/8)
# c5, c6 ~ d2
# c7 = c6 in 7/8
# c8 = n3 < 0 in 7/8
# c9 = c6 xor c8 in 7/8
# class = [c7 and (n2 + n3 <  0.5)] or [!c7 and (n3 + n4 < -0.5)] in 31/32
# n0
# n1 - In-Directly related to class label : Yes
# n3 - Directly related to class label : Yes
# n2 - Directly related to class label : Yes
# n4 - Directly related to class label : Yes
# c5
# c6 - In-Directly related to class label : Yes
# c7 - Directly related to class label : Yes
# c8
# c9


def detect_concept_drift(
    model, X_df, y_df, metric_func, drift_detector, window_size=100
):
    timepoint = 0
    metric_score_list = []  # Record the real-time metric
    concept_drifts_timepoints = []
    my_model = model
    my_drift_detector = drift_detector

    window_y = deque(maxlen=window_size)

    # Stream data, updating the metric and checking for drifts
    for xi, yi in stream.iter_pandas(X_df, y_df):
        y_pred = my_model.predict_one(xi)

        # Update the window with new prediction and get the current metric value of the window
        window_y.append((yi, y_pred))
        windowed_metric_val = metric_func(zip(*window_y)) * 100

        my_model.learn_one(xi, yi)

        # Recalculate the metric using only the values within the window
        if len(window_y) == window_size:

            my_drift_detector.update(windowed_metric_val)

            if my_drift_detector.drift_detected:
                concept_drifts_timepoints.append(timepoint)
                my_model = my_model.clone()  # Clone the model upon drift detection
                window_y.clear()  # Clear window on drift detection

        metric_score_list.append(windowed_metric_val)

        timepoint += 1

    return concept_drifts_timepoints, metric_score_list


from sklearn.metrics import accuracy_score

z = detect_concept_drift(HoeffdingTreeClassifier(), X_df, y_df, accuracy_score, ADWIN())
