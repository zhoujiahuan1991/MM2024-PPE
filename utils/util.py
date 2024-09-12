import torch.distributed as dist
import numpy as np
from scipy.stats import sem
import scipy.stats as stats

def compute_performance(end_task_acc_arr):
    """
    Given test accuracy results from multiple runs saved in end_task_acc_arr,
    compute the average accuracy, forgetting, and task accuracies as well as their confidence intervals.

    :param end_task_acc_arr:       (list) List of lists
    :param task_ids:                (list or tuple) Task ids to keep track of
    :return:                        (avg_end_acc, forgetting, avg_acc_task)
    """
    n_run, n_tasks = end_task_acc_arr.shape[:2]
    t_coef = stats.t.ppf((1+0.95) / 2, n_run-1)     # t coefficient used to compute 95% CIs: mean +- t *

    # compute average test accuracy and CI
    end_acc = end_task_acc_arr[:, -1, :]                         # shape: (num_run, num_task)
    avg_acc_per_run = np.mean(end_acc, axis=1)      # mean of end task accuracies per run
    avg_end_acc = (np.mean(avg_acc_per_run), t_coef * sem(avg_acc_per_run))

    # compute forgetting
    best_acc = np.max(end_task_acc_arr, axis=1)
    final_forgets = best_acc - end_acc
    avg_fgt = np.mean(final_forgets, axis=1)
    avg_end_fgt = (np.mean(avg_fgt), t_coef * sem(avg_fgt))

    return avg_end_acc, avg_end_fgt
