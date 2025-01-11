import numpy as np

def exponential_schedule(initial_value: float, numer_of_steps: int, exponent: float):
    """
    stepwise learning rate schedule.
    We start at initial learning rate,
    and reduce the learning rate by 50% every 1/numer_of_steps of the total number of steps (as indicated by progress remaining)

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    list_of_multipliers = [1]
    for i in range(numer_of_steps-1):
        list_of_multipliers.append(list_of_multipliers[-1] * exponent)
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end).
        We start at initial learning rate,
        and reduce the learning rate by 50% every 1/numer_of_steps of the total number of steps (as indicated by progress remaining)

        :param progress_remaining:
        :return: current learning rate
        """
        mutliplier_index = numer_of_steps - int(np.ceil(progress_remaining * numer_of_steps))
        return list_of_multipliers[mutliplier_index] * initial_value

    return func