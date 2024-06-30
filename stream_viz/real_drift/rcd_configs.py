from river.linear_model import LogisticRegression
from river.tree import HoeffdingTreeClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from stream_viz.real_drift.mddm import MDDM_A, MDDM_E, MDDM_G

metrics_dict = {
    "Accuracy": accuracy_score,
    "Cohen Kappa": cohen_kappa_score,
    "F1 Score": f1_score,
}

models_dict = {
    "Hoeffding": HoeffdingTreeClassifier(),
    "Logistic": LogisticRegression(),
}

drift_detectors = {"MDDM_A": MDDM_A, "MDDM_E": MDDM_E, "MDDM_G": MDDM_G}


if __name__ == "__main__":
    # model = HoeffdingTreeClassifier()
    pass
