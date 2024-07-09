# Reference:
# Pesaranghader, A., Viktor, H.L., & Paquet, E. (2017).
# McDiarmid Drift Detection Methods for Evolving Data Streams. 2018
# International Joint Conference on Neural Networks (IJCNN), 1-9.

import math


class MDDM_A:
    def __init__(
        self,
        sliding_win_size=100,
        difference=0.01,
        confidence=0.000001,
        warning_confidence=0.000005,
    ):
        self.sliding_win_size = sliding_win_size
        self.difference = difference
        self.confidence = confidence
        self.warning_confidence = warning_confidence

        self.win = [0] * self.sliding_win_size
        self.pointer = 0

        self.delta = self.confidence
        self.epsilon = (0.5 * self.cal_sigma() * (math.log(1 / self.delta))) ** 0.5
        self.warning_delta = self.warning_confidence
        self.warning_epsilon = (
            0.5 * self.cal_sigma() * (math.log(1 / self.warning_delta))
        ) ** 0.5
        self.u_max = 0

        self.is_change_detected = False
        self.is_initialized = False
        self.is_warning_zone = False

    def reset_learning(self):
        self.win = [0] * self.sliding_win_size
        self.pointer = 0
        self.difference = self.difference
        self.delta = self.confidence
        self.epsilon = (0.5 * self.cal_sigma() * (math.log(1 / self.delta))) ** 0.5
        self.warning_delta = self.warning_confidence
        self.warning_epsilon = (
            0.5 * self.cal_sigma() * (math.log(1 / self.warning_delta))
        ) ** 0.5
        self.u_max = 0

    def input(self, prediction):
        if self.is_change_detected or not self.is_initialized:
            self.reset_learning()
            self.is_initialized = True

        drift_status = False
        warning_status = False

        if self.pointer < len(self.win):
            self.win[self.pointer] = 1 if prediction == 0 else 0
            self.pointer += 1
        else:
            for i in range(len(self.win)):
                if i == len(self.win) - 1:
                    self.win[i] = 1 if prediction == 0 else 0
                else:
                    self.win[i] = self.win[i + 1]

        if self.pointer == len(self.win):
            u = self.cal_w_mean()
            self.u_max = max(self.u_max, u)
            if self.u_max - u > self.epsilon:
                drift_status = True
            elif self.u_max - u > self.warning_epsilon:
                warning_status = True

        self.is_warning_zone = warning_status
        self.is_change_detected = drift_status

    def cal_sigma(self):
        total_sum = sum(1 + i * self.difference for i in range(len(self.win)))
        return sum(
            ((1 + i * self.difference) / total_sum) ** 2 for i in range(len(self.win))
        )

    def cal_w_mean(self):
        total_sum = sum(1 + i * self.difference for i in range(len(self.win)))
        win_sum = sum(
            self.win[i] * (1 + i * self.difference) for i in range(len(self.win))
        )
        return win_sum / total_sum

    def get_description(self, indent=0):
        pass

    def prepare_for_use(self, monitor, repository):
        pass


class MDDM_E:
    def __init__(
        self,
        sliding_win_size=100,
        lambda_val=0.01,
        confidence=0.000001,
        warning_confidence=0.000005,
    ):
        self.sliding_win_size = sliding_win_size
        self.lambda_val = lambda_val
        self.confidence = confidence
        self.warning_confidence = warning_confidence

        self.win = [0] * self.sliding_win_size
        self.pointer = 0

        self.lambda_val = self.lambda_val
        self.delta = self.confidence
        self.epsilon = math.sqrt(0.5 * self.cal_sigma() * math.log(1 / self.delta))
        self.u_max = 0
        self.warning_delta = self.warning_confidence
        self.warning_epsilon = (
            0.5 * self.cal_sigma() * (math.log(1 / self.warning_delta))
        ) ** 0.5

        self.is_change_detected = False
        self.is_initialized = False
        self.is_warning_zone = False

    def reset_learning(self):
        self.win = [0] * self.sliding_win_size
        self.pointer = 0
        self.lambda_val = self.lambda_val
        self.delta = self.confidence
        self.epsilon = math.sqrt(0.5 * self.cal_sigma() * math.log(1 / self.delta))
        self.warning_delta = self.warning_confidence
        self.warning_epsilon = (
            0.5 * self.cal_sigma() * (math.log(1 / self.warning_delta))
        ) ** 0.5
        self.u_max = 0

    def input(self, prediction):
        if self.is_change_detected or not self.is_initialized:
            self.reset_learning()
            self.is_initialized = True

        drift_status = False
        warning_status = False

        if self.pointer < len(self.win):
            self.win[self.pointer] = 1 if prediction == 0 else 0
            self.pointer += 1
        else:
            for i in range(len(self.win) - 1):
                self.win[i] = self.win[i + 1]
            self.win[-1] = 1 if prediction == 0 else 0

        if self.pointer == len(self.win):
            u = self.cal_w_mean()
            self.u_max = max(self.u_max, u)
            if self.u_max - u > self.epsilon:
                drift_status = True
            elif self.u_max - u > self.warning_epsilon:
                warning_status = True
        self.is_warning_zone = warning_status
        self.is_change_detected = drift_status

    def cal_sigma(self):
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

    def cal_w_mean(self):
        total_sum = 0
        win_sum = 0
        r = 1
        ratio = math.exp(self.lambda_val)
        for i in range(len(self.win)):
            total_sum += r
            win_sum += self.win[i] * r
            r *= ratio
        return win_sum / total_sum

    def get_description(self, indent=0):
        pass

    def prepare_for_use(self, monitor, repository):
        pass


class MDDM_G:
    def __init__(
        self,
        sliding_win_size=100,
        ratio=1.01,
        confidence=0.000001,
        warning_confidence=0.000005,
    ):
        self.sliding_win_size = sliding_win_size
        self.ratio = ratio
        self.confidence = confidence
        self.warning_confidence = warning_confidence

        self.win = [0] * self.sliding_win_size
        self.pointer = 0

        self.ratio = self.ratio
        self.delta = self.confidence
        self.epsilon = math.sqrt(0.5 * self.cal_sigma() * math.log(1 / self.delta))
        self.warning_delta = self.warning_confidence
        self.warning_epsilon = (
            0.5 * self.cal_sigma() * (math.log(1 / self.warning_delta))
        ) ** 0.5
        self.u_max = 0

        self.is_change_detected = False
        self.is_initialized = False
        self.is_warning_zone = False

    def reset_learning(self):
        self.win = [0] * self.sliding_win_size
        self.pointer = 0
        self.ratio = self.ratio
        self.delta = self.confidence
        self.epsilon = math.sqrt(0.5 * self.cal_sigma() * math.log(1 / self.delta))
        self.warning_delta = self.warning_confidence
        self.warning_epsilon = (
            0.5 * self.cal_sigma() * (math.log(1 / self.warning_delta))
        ) ** 0.5
        self.u_max = 0

    def input(self, prediction):
        if self.is_change_detected or not self.is_initialized:
            self.reset_learning()
            self.is_initialized = True

        drift_status = False
        warning_status = False

        if self.pointer < len(self.win):
            self.win[self.pointer] = 1 if prediction == 0 else 0
            self.pointer += 1
        else:
            for i in range(len(self.win) - 1):
                self.win[i] = self.win[i + 1]
            self.win[-1] = 1 if prediction == 0 else 0

        if self.pointer == len(self.win):
            u = self.cal_w_mean()
            self.u_max = max(self.u_max, u)
            if self.u_max - u > self.epsilon:
                drift_status = True
            elif self.u_max - u > self.warning_epsilon:
                warning_status = True

        self.is_warning_zone = warning_status
        self.is_change_detected = drift_status

    def cal_sigma(self):
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

    def cal_w_mean(self):
        total_sum = 0
        win_sum = 0
        r = self.ratio
        for i in range(len(self.win)):
            total_sum += r
            win_sum += self.win[i] * r
            r *= self.ratio
        return win_sum / total_sum
