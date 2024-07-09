import math
from abc import ABC, abstractmethod
from collections import deque


class _MDDMBase(ABC):
    """
    Abstract base class for McDiarmid Drift Detection Method (MDDM) models.

    References:
        Pesaranghader, A., Viktor, H.L., & Paquet, E. (2017).
        McDiarmid Drift Detection Methods for Evolving Data Streams. 2018
        International Joint Conference on Neural Networks (IJCNN), 1-9.

    Attributes:
        sliding_win_size (int): Size of the sliding window for monitoring.
        confidence (float): Confidence level for drift detection.
        warning_confidence (float): Confidence level for warning zone detection.
        win (deque): Circular buffer to store recent predictions.
        pointer (int): Current index in the circular buffer.
        delta (float): Confidence level for drift detection.
        epsilon (float): Threshold for detecting drift.
        warning_delta (float): Confidence level for warning zone detection.
        warning_epsilon (float): Threshold for entering warning zone.
        u_max (float): Maximum weighted mean encountered.
        is_change_detected (bool): Flag indicating if drift is detected.
        is_initialized (bool): Flag indicating if the model is initialized.
        is_warning_zone (bool): Flag indicating if warning zone is active.
    """

    def __init__(
        self,
        sliding_win_size: int = 100,
        confidence: float = 0.000001,
        warning_confidence: float = 0.000005,
    ) -> None:
        """
        Initialize the MDDM base class with specified parameters.

        Args:
            sliding_win_size (int, optional): Size of the sliding window (default: 100).
            confidence (float, optional): Confidence level for drift detection (default: 0.000001).
            warning_confidence (float, optional): Confidence level for warning zone detection (default: 0.000005).
        """
        self.sliding_win_size = sliding_win_size
        self.confidence = confidence
        self.warning_confidence = warning_confidence

        self.win = deque([0] * self.sliding_win_size)
        self.pointer = 0

        self.delta = self.confidence
        self.epsilon = 0.0
        self.warning_delta = self.warning_confidence
        self.warning_epsilon = 0.0
        self.u_max = 0.0

        self.is_change_detected = False
        self.is_initialized = False
        self.is_warning_zone = False

    def reset_learning(self) -> None:
        """
        Reset the learning state of the model.
        """
        self.win = deque([0] * self.sliding_win_size)
        self.pointer = 0
        self.delta = self.confidence
        self.epsilon = math.sqrt(0.5 * self.cal_sigma() * math.log(1 / self.delta))
        self.warning_delta = self.warning_confidence
        self.warning_epsilon = math.sqrt(
            0.5 * self.cal_sigma() * math.log(1 / self.warning_delta)
        )
        self.u_max = 0.0

    def input(self, prediction: int) -> None:
        """
        Process a new prediction and update the drift detection state.

        Args:
            prediction (int): Predicted class label (0 or 1).
        """
        if self.is_change_detected or not self.is_initialized:
            self.reset_learning()
            self.is_initialized = True

        drift_status = False
        warning_status = False

        self.win.popleft()
        self.win.append(1 if prediction == 0 else 0)

        if self.pointer == self.sliding_win_size:
            u = self.cal_w_mean()
            self.u_max = max(self.u_max, u)
            if self.u_max - u > self.epsilon:
                drift_status = True
            elif self.u_max - u > self.warning_epsilon:
                warning_status = True

        self.is_warning_zone = warning_status
        self.is_change_detected = drift_status

    @abstractmethod
    def cal_sigma(self) -> float:
        """
        Abstract method to calculate sigma, to be implemented by subclasses.

        Returns:
            float: Calculated sigma value.
        """
        raise NotImplementedError("Subclasses should implement cal_sigma method")

    @abstractmethod
    def cal_w_mean(self) -> float:
        """
        Abstract method to calculate weighted mean, to be implemented by subclasses.

        Returns:
            float: Calculated weighted mean.
        """
        raise NotImplementedError("Subclasses should implement cal_w_mean method")


class MDDM_A(_MDDMBase):
    """
    MDDM model based on Arithmetic weighting method for drift detection.

    Arithmetic Weighting Scheme:
        Arithmetic : w_i = 1 + ( i + d) ...where d ≥ 0, is difference between two consecutive weights

    Inherits from _MDDMBase.
    """

    def __init__(
        self,
        sliding_win_size: int = 100,
        difference: float = 0.01,
        confidence: float = 0.000001,
        warning_confidence: float = 0.000005,
    ) -> None:
        """
        Initialize the MDDM_A model with specified parameters.

        Args:
            sliding_win_size (int, optional): Size of the sliding window (default: 100).
            difference (float, optional): Difference parameter for A-statistic calculation (default: 0.01).
            confidence (float, optional): Confidence level for drift detection (default: 0.000001).
            warning_confidence (float, optional): Confidence level for warning zone detection (default: 0.000005).
        """
        super().__init__(sliding_win_size, confidence, warning_confidence)
        self.difference = difference

    def reset_learning(self) -> None:
        """
        Reset the learning state of the MDDM_A model.
        """
        super().reset_learning()
        self.epsilon = math.sqrt(0.5 * self.cal_sigma() * math.log(1 / self.delta))
        self.warning_epsilon = math.sqrt(
            0.5 * self.cal_sigma() * math.log(1 / self.warning_delta)
        )
        self.u_max = 0.0

    def cal_sigma(self) -> float:
        """
        Calculate the sigma value for Arithmetic weighting method.

        Returns:
            float: Calculated sigma value.
        """
        total_sum = sum(1 + i * self.difference for i in range(len(self.win)))
        return sum(
            ((1 + i * self.difference) / total_sum) ** 2 for i in range(len(self.win))
        )

    def cal_w_mean(self) -> float:
        """
        Calculate the weighted mean for Arithmetic weighting method.

        Returns:
            float: Calculated weighted mean.
        """
        total_sum = sum(1 + i * self.difference for i in range(len(self.win)))
        win_sum = sum(
            self.win[i] * (1 + i * self.difference) for i in range(len(self.win))
        )
        return win_sum / total_sum


class MDDM_E(_MDDMBase):
    """
    MDDM model based on Euler Weighting method for drift detection.

    Euler Weighting Scheme:
        Euler: w_i = r^(i-1)   ..with r = e^lambda ... where lambda ≥ 0

    Inherits from _MDDMBase.
    """

    def __init__(
        self,
        sliding_win_size: int = 100,
        lambda_val: float = 0.01,
        confidence: float = 0.000001,
        warning_confidence: float = 0.000005,
    ) -> None:
        """
        Initialize the MDDM_E model with specified parameters.

        Args:
            sliding_win_size (int, optional): Size of the sliding window (default: 100).
            lambda_val (float, optional): Lambda value for Euler Weighting calculation (default: 0.01).
            confidence (float, optional): Confidence level for drift detection (default: 0.000001).
            warning_confidence (float, optional): Confidence level for warning zone detection (default: 0.000005).
        """
        super().__init__(sliding_win_size, confidence, warning_confidence)
        self.lambda_val = lambda_val

    def reset_learning(self) -> None:
        """
        Reset the learning state of the MDDM_E model.
        """
        super().reset_learning()
        self.epsilon = math.sqrt(0.5 * self.cal_sigma() * math.log(1 / self.delta))
        self.warning_epsilon = math.sqrt(
            0.5 * self.cal_sigma() * math.log(1 / self.warning_delta)
        )
        self.u_max = 0.0

    def cal_sigma(self) -> float:
        """
        Calculate the sigma value for Euler Weighting method.

        Returns:
            float: Calculated sigma value.
        """
        sum_exp = 0
        bound_sum = 0
        r = 1
        ratio = math.exp(self.lambda_val)
        for i in range(len(self.win)):
            sum_exp += r
            r *= ratio
        r = 1
        for i in range(len(self.win)):
            bound_sum += (r / sum_exp) ** 2
            r *= ratio
        return bound_sum

    def cal_w_mean(self) -> float:
        """
        Calculate the weighted mean for Euler Weighting method.

        Returns:
            float: Calculated weighted mean.
        """
        total_sum = 0
        win_sum = 0
        r = 1
        ratio = math.exp(self.lambda_val)
        for i in range(len(self.win)):
            total_sum += r
            win_sum += self.win[i] * r
            r *= ratio
        return win_sum / total_sum


class MDDM_G(_MDDMBase):
    """
    MDDM model based on Geometric Weighting method for drift detection.

    Geometric Weighting Scheme:
        Geometric : w_i = r^(i-1) ...where r ≥ 1, is ratio between two consecutive weights

    Inherits from _MDDMBase.
    """

    def __init__(
        self,
        sliding_win_size: int = 100,
        ratio: float = 1.01,
        confidence: float = 0.000001,
        warning_confidence: float = 0.000005,
    ) -> None:
        """
        Initialize the MDDM_G model with specified parameters.

        Args:
            sliding_win_size (int, optional): Size of the sliding window (default: 100).
            ratio (float, optional): Ratio parameter for G-statistic calculation (default: 1.01).
            confidence (float, optional): Confidence level for drift detection (default: 0.000001).
            warning_confidence (float, optional): Confidence level for warning zone detection (default: 0.000005).
        """
        super().__init__(sliding_win_size, confidence, warning_confidence)
        self.ratio = ratio

    def reset_learning(self) -> None:
        """
        Reset the learning state of the MDDM_G model.
        """
        super().reset_learning()
        self.epsilon = math.sqrt(0.5 * self.cal_sigma() * math.log(1 / self.delta))
        self.warning_epsilon = math.sqrt(
            0.5 * self.cal_sigma() * math.log(1 / self.warning_delta)
        )
        self.u_max = 0.0

    def cal_sigma(self) -> float:
        """
        Calculate the sigma value for Geometric Weighting method.

        Returns:
            float: Calculated sigma value.
        """
        sum_exp = 0
        bound_sum = 0
        r = self.ratio
        for i in range(len(self.win)):
            sum_exp += r
            r *= self.ratio
        r = self.ratio
        for i in range(len(self.win)):
            bound_sum += (r / sum_exp) ** 2
            r *= self.ratio
        return bound_sum

    def cal_w_mean(self) -> float:
        """
        Calculate the weighted mean for Geometric Weighting method.

        Returns:
            float: Calculated weighted mean.
        """
        total_sum = 0
        win_sum = 0
        r = self.ratio
        for i in range(len(self.win)):
            total_sum += r
            win_sum += self.win[i] * r
            r *= self.ratio
        return win_sum / total_sum


if __name__ == "__main__":
    # Example usage
    mddm_a = MDDM_A()
    mddm_e = MDDM_E()
    mddm_g = MDDM_G()

    # Simulating input predictions
    predictions = [0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
    for pred in predictions:
        mddm_a.input(pred)
        mddm_e.input(pred)
        mddm_g.input(pred)

    # Checking drift and warning statuses
    print(
        f"MDDM_A: Change detected: {mddm_a.is_change_detected}, Warning zone: {mddm_a.is_warning_zone}"
    )
    print(
        f"MDDM_E: Change detected: {mddm_e.is_change_detected}, Warning zone: {mddm_e.is_warning_zone}"
    )
    print(
        f"MDDM_G: Change detected: {mddm_g.is_change_detected}, Warning zone: {mddm_g.is_warning_zone}"
    )
