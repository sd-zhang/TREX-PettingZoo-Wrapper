from typing import Tuple

import numpy as np


class RunningMeanStdMinMax:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = (),
                 forced_maxes: np.ndarray = None, forced_mins: np.ndarray = None):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        if forced_maxes is None:
            forced_maxes = np.inf * np.ones(shape, np.float64)
        assert forced_maxes.shape == shape, "Forced maxes shape does not match shape"
        if forced_mins is None:
            forced_mins = -np.inf * np.ones(shape, np.float64)
        assert forced_mins.shape == shape, "Forced mins shape does not match shape"
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon
        self.max = -np.inf * np.ones(shape, np.float64)
        self.force_maxes = forced_maxes
        self.min = np.inf * np.ones(shape, np.float64)
        self.force_mins = forced_mins

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStdMinMax(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStdMinMax") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_max = np.max(arr, axis=0)
        batch_min = np.min(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_max, batch_min, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_max: np.ndarray, batch_min: np.ndarray,
                            batch_count: float) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
        self.max = np.maximum(self.max, batch_max)
        # set the max to all the forced maxes that are not None
        self.max = np.where(self.force_maxes is None, self.force_maxes, self.max)

        self.min = np.minimum(self.min, batch_min)
        # set the min to all the forced mins that are not None
        self.min = np.where(self.force_mins is None, self.force_mins, self.min)