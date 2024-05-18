from river import base
from river.drift.retrain import DriftRetrainingClassifier


class DriftRetrainClassifier(DriftRetrainingClassifier):
    def __init__(
        self,
        model: base.Classifier,
        drift_detector: (
            base.DriftAndWarningDetector | base.BinaryDriftAndWarningDetector | None
        ) = None,
        train_in_background: bool = True,
    ):
        super(DriftRetrainClassifier, self).__init__(
            model, drift_detector, train_in_background
        )

        self.drift_detection_timepoints = []

    @property
    def drift_timepoints(self):
        return self.drift_detection_timepoints

    def learn_one(self, x, y, **kwargs):
        # Check if 'curr_timepoint' is present in kwargs
        if "curr_timepoint" not in kwargs:
            raise ValueError("Missing required parameter: 'curr_timepoint'")

        # Check if 'curr_timepoint' is an integer
        if not isinstance(kwargs["curr_timepoint"], int):
            raise TypeError("Parameter 'curr_timepoint' must be of type int")

        self._update_detector(x, y, **kwargs)
        self.model.learn_one(x, y)

    def _update_detector(self, x, y, **kwargs):
        y_pred = self.model.predict_one(x)
        if y_pred is None:
            return

        incorrectly_classifies = int(y_pred != y)
        self.drift_detector.update(incorrectly_classifies)

        curr_timepoint = int(kwargs.get("curr_timepoint"))

        if self.train_in_background:
            if self.drift_detector.warning_detected:
                # If there's a warning, we train the background model
                self.bkg_model.learn_one(x, y)
            elif self.drift_detector.drift_detected:
                # If there's a drift, we replace the model with the background model
                self.model = self.bkg_model
                self.bkg_model = self.model.clone()
                self.drift_detection_timepoints.append(curr_timepoint)
        else:
            if self.drift_detector.drift_detected:
                # If there's a drift, we reset the model
                self.model = self.model.clone()
                self.drift_detection_timepoints.append(curr_timepoint)
