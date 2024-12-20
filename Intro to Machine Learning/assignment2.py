#################################
# Your name: Idan Drori
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
            A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        x_samples = np.random.uniform(0, 1, m)
        x_samples.sort()
        y_samples = np.array([np.random.choice([0, 1], p=[self.comp_distribution(x), self.true_distribution(x)])
                              for x in x_samples])
        return np.column_stack((x_samples, y_samples))

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        n_steps = (m_last - m_first) // step + 1  # Number of steps
        results = np.zeros((n_steps, 2))
        print(f"Running ERM with k={k}, for {n_steps} steps")
        for i, m in enumerate(range(m_first, m_last + 1, step)):
            print(f"Running on {m} samples")
            empirical_errors = []
            true_errors = []
            for t in range(T):
                sample = self.sample_from_D(m)
                emp_error, true_error = self.get_errors(sample, k, m)
                empirical_errors.append(emp_error)
                true_errors.append(true_error)

            avg_emp_error = np.mean(empirical_errors)
            avg_true_error = np.mean(true_errors)
            results[i] = [avg_emp_error, avg_true_error]

        plt.plot(range(m_first, m_last + 1, step), results[:, 0], label="Empirical Error")
        plt.plot(range(m_first, m_last + 1, step), results[:, 1], label="True Error")
        plt.xlabel("n")
        plt.ylabel("Error")
        plt.title("Average Error as a Function of n")
        plt.legend()
        plt.show()

        return results

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        k_values = list(range(k_first, k_last + 1, step))
        empirical_errors = []
        true_errors = []
        print(f"Running ERM with m={m}, k running from {k_first} to {k_last} with step={step}")
        sample = self.sample_from_D(m)
        for k in k_values:
            print(f"Running on k={k}")
            emp_error, true_error = self.get_errors(sample, k, m)
            empirical_errors.append(emp_error)
            true_errors.append(true_error)

        plt.plot(k_values, empirical_errors, label="Empirical Error")
        plt.plot(k_values, true_errors, label="True Error")
        plt.xlabel("k")
        plt.ylabel("Error")
        plt.title("Empirical and True Errors as a Function of k")
        plt.legend()
        plt.show()

        best_k_idx = np.argmin(true_errors)
        best_k = k_values[best_k_idx]
        return best_k

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        delta = 0.1
        empirical_errors = []
        true_errors = []
        penalties = []
        sum_errors = []
        k_values = list(range(k_first, k_last + 1, step))

        print(f"Running ERM with SRM penalty with m={m}, k's running from {k_first} to {k_last} with step={step}")
        sample = self.sample_from_D(m)
        for k in k_values:
            print(f"Running on k={k}")
            emp_error, true_error = self.get_errors(sample, k, m)
            empirical_errors.append(emp_error)
            true_errors.append(true_error)
            # SRM
            penalty = 2 * np.sqrt((2 * k + np.log(2 / delta)) / m)  # SRM Penalty, VCdim(H) = 2k
            penalties.append(penalty)
            sum_errors.append(emp_error + penalty)

        plt.plot(k_values, empirical_errors, label="Empirical Error")
        plt.plot(k_values, true_errors, label="True Error")
        plt.plot(k_values, penalties, label="Penalty")
        plt.plot(k_values, sum_errors, label="Penalty + Empirical Error")
        plt.xlabel("k")
        plt.ylabel("Error")
        plt.title("Errors, Penalty, and Penalty + Empirical Error as Functions of k")
        plt.legend()
        plt.show()

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        holdout = 0.2
        num_holdout = int(m * holdout)
        training_set = self.sample_from_D(m - num_holdout)
        validation_set = self.sample_from_D(num_holdout)

        errors = []
        hypotheses = []
        print(f"Running cross validation with m={m}")
        for k in range(1, 11):
            print(f"Running on k={k}")
            best_intervals, useless = intervals.find_best_interval(training_set[:, 0], training_set[:, 1], k)
            hypotheses.append(best_intervals)
            emp_error = self.calc_empirical_error(best_intervals, validation_set)
            errors.append(emp_error)

        best_k = np.argmin(errors) + 1
        print(f"Best value of k is {best_k} with a true error of {errors[best_k - 1]}")
        print(f"Best hypothesis is {hypotheses[best_k - 1]}")
        return best_k


    #################################
    # Place for additional methods
    #################################

    def true_distribution(self, x):
        """
        Returns the probability of y=1 given x, according to the given distribution
        :param x: float between [0,1]
        :return: Probability of y=1 given x.
        :rtype: float
        """
        if 0 <= x <= 0.2 or 0.4 <= x <= 0.6 or 0.8 <= x <= 1:
            return 0.8
        elif 0.2 < x < 0.4 or 0.6 < x < 0.8:
            return 0.1
        else:  # Will only get here if given an x not in [0,1]
            return -1

    def comp_distribution(self, x):
        """
        Returns the probability of y=0 given x, according to the given distribution
        :param x: float between [0,1]
        :return: Probability of y=0 given x
        :rtype: float
        """
        if 0 <= x <= 1:
            return 1 - self.true_distribution(x)
        else:  # Should only get here if x not in [0,1]
            return -1

    def get_intersect(self, interval1, interval2):
        """
        Get the intersection of two intervals
        :param interval1: A tuple of two floats, first is the left bound and second is the right bound
        :param interval2: A tuple of two floats, first is the left bound and second is the right bound
        :return: The length of the intersection
        """
        return min(interval1[1], interval2[1]) - max(interval1[0], interval2[0])

    def comp_intervals(self, intervals):
        """
        Get the complement of a list of intervals. This assumes the list of intervals is ordered
        :param intervals: A list of tuples, where each tuple is a left and right bound of an interval
        :return: The complement of the intervals
        """
        complement_intervals = []
        if intervals[0][0] > 0:  # If the first interval doesn't start at 0,
            # add an interval from 0 to the start of the first interval.
            complement_intervals.append([0, intervals[0][0]])
        for i in range(len(intervals) - 1):  # Add every interval that's between two intervals in the list
            complement_intervals.append([intervals[i][1], intervals[i + 1][0]])
        if intervals[-1][1] < 1:  # If the final interval doesn't stop at 1,
            # add an interval from the end of the last interval up to 1
            complement_intervals.append([intervals[-1][1], 1])
        return complement_intervals

    def calc_true_error(self, best_intervals):
        """
        Calculate the true error of the best hypothesis
        :param best_intervals: A list of tuples, each tuple has the left and right bounds of an interval
        :return: The true error
        """
        total_error = 0
        # List of tuples, where each tuple is an interval, and the cost we need to pay for classifying as 0.
        positive_intervals = [([0, 0.2], 0.2), ([0.4, 0.6], 0.2), ([0.8, 1], 0.2), ([0.2, 0.4], 0.9), ([0.6, 0.8], 0.9)]
        # Iterating over the intervals and costs (h(x)=1)
        for interval, miss_prob in positive_intervals:
            for best_interval in best_intervals:
                intersection = self.get_intersect(interval, best_interval)
                if intersection > 0:
                    # We need to pay for the length of the interval multiplied by P(y=0|x)
                    total_error += intersection * miss_prob

        # Complement of best intervals
        best_intervals_complement = self.comp_intervals(best_intervals)
        # List of tuples, where each tuple is an interval and the cost we need to pay for classifying as 1.
        negative_intervals = [([0, 0.2], 0.8), ([0.4, 0.6], 0.8), ([0.8, 1], 0.8), ([0.2, 0.4], 0.1), ([0.6, 0.8], 0.1)]
        # Iterating over the intervals and costs (h(x)=0)
        for interval, miss_prob in negative_intervals:
            for best_interval_comp in best_intervals_complement:
                intersection = self.get_intersect(interval, best_interval_comp)
                if intersection > 0:
                    # We need to pay for the length of the interval multiplied by P(y=1|x)
                    total_error += intersection * miss_prob
        return total_error

    def get_errors(self, sample, k, m):
        """
        Get the empirical and true errors of the best hypothesis, gotten by running find_best_interval.
        :param sample: A two-dimensional array of size m that contains the x,y pairs from the distribution P
        :param k: An integer, the maximum number of intervals
        :param m: An integer, the size of the data
        :return: A tuple of two floats, the empirical error on [0] and the true error on [1]
        """
        best_intervals, empirical_error = intervals.find_best_interval(sample[:, 0], sample[:, 1], k)
        true_error = self.calc_true_error(best_intervals)
        return empirical_error / m, true_error

    def calc_empirical_error(self, best_intervals, test_set):
        """
        Calculate the empirical error of the best hypothesis
        :param best_intervals: A list of tuples, where each tuple is the left and right bound of an interval
        :param test_set: A two-dimensional array of size m that contains the pairs that where drawn from P
        :return: The empirical error
        """
        mislabel_count = 0
        for x, y in test_set:
            # Check to see if x falls in any of the intervals
            in_interval = any(left <= x <= right for left, right in best_intervals)
            if (in_interval and y == 0) or (not in_interval and y == 1):
                # If x was mislabeled we update the count
                mislabel_count += 1
        return mislabel_count / len(test_set)


if __name__ == '__main__':
    ass = Assignment2()
    #ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    #ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)

