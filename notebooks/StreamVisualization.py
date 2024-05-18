import matplotlib.pyplot as plt
import mplcursors
import mpld3
import numpy as np
import pandas as pd
import seaborn as sns
from DriftRetrainClassifier import DriftRetrainClassifier
from river import linear_model, stream, tree
from river.drift import ADWIN, KSWIN, DriftRetrainingClassifier, PageHinkley
from river.drift.binary import *
from river.forest import ARFClassifier
from river.linear_model import LogisticRegression
from river.metrics import F1, Accuracy, BalancedAccuracy, CohenKappa, Precision, Recall
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


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
    EFDTC_model = tree.ExtremelyFastDecisionTreeClassifier()
    # model  = EFDTC_model
    arf_model = ARFClassifier()
    lev_bag_model = LeveragingBaggingClassifier()
    # model = tree.hoeffding_adaptive_tree_classifier.HoeffdingAdaptiveTreeClassifier()
    # model = ADWINBoostingClassifier()
    # model = ADWINBaggingClassifier()
    hoeffding_model = tree.HoeffdingTreeClassifier()

    model = DriftRetrainClassifier(
        model=hoeffding_model,
        # drift_detector=drift.binary.DDM()
        drift_detector=ADWIN(),
        train_in_background=False,
    )

    # t, m = stream_viz.adaptive_learning(model,
    #                                     X_train,
    #                                     y_train,
    #                                     X_test,
    #                                     y_test,
    #                                     metrics[0]())
    logistic_model = linear_model.LogisticRegression()

    # t, m = stream_viz.adaptive_learning_2(hoeffding_model,
    #                                     X_train,
    #                                     y_train,
    #                                     X_test,
    #                                     y_test,
    #                                     CohenKappa())

    t, m = stream_viz.adaptive_learning_3(hoeffding_model, X_df, y_df, CohenKappa())
    stream_viz.acc_fig(t, m, "sdas")
    # stream_ = StreamVisualization(X_df, y_df)
    # stream_.detect_concept_drift()
