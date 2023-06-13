from scipy import stats
import torch

def unfairness_re(data1, data2):
    """
    compute the unfairness of two populations
    """
    unfair_value = stats.ks_2samp(data1, data2, alternative='two-sided')[0]
    return unfair_value

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)