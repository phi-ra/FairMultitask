"""
Module for fairness estimates
"""
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np

class EQF:
    def __init__(self, 
                 sample_data):
        self._calculate_eqf(sample_data)

    def _calculate_eqf(self,sample_data):
        sorted_data = np.sort(sample_data)
        linspace  = np.linspace(0,1,num=len(sample_data))
        self.interpolater = interp1d(linspace, sorted_data)
        self.min_val = sorted_data[0]
        self.max_val = sorted_data[-1]

    def __call__(self, value_):
        try:
            return self.interpolater(value_)
        except ValueError:
            if value_ < self.min_val:
                return 0.0
            elif value_ > self.max_val:
                return 1.0
            else:
                raise ValueError('Error with input value')
            
def get_fairness_objects(sensitive_vector,
                         predictions_s_0, 
                         predictions_s_1):
    
    # calculate CDF and quantile function
    ecdf_dict = {}
    eqf_dict = {}
    pi_dict = {}

    ecdf_dict['p_non_sensitive'] = ECDF(predictions_s_0.reshape(-1,))
    ecdf_dict['p_sensitive'] = ECDF(predictions_s_1.reshape(-1,))

    eqf_dict['p_non_sensitive'] = EQF(predictions_s_0.reshape(-1,))
    eqf_dict['p_sensitive'] = EQF(predictions_s_1.reshape(-1,))

    pi_dict['p_non_sensitive'] = sensitive_vector[sensitive_vector == 0.0].shape[0] / sensitive_vector.shape[0]
    pi_dict['p_sensitive'] = 1-pi_dict['p_non_sensitive']

    return pi_dict, ecdf_dict, eqf_dict


def get_fair_estimation(p_dict, 
                        ecdf_dict, 
                        eqf_dict,
                        predictions_nonsensitve, 
                        predictions_sensitve,
                        jitter=0.0001, 
                        seed=42):
    
    # sample jitters
    np.random.seed(seed)
    jitter_matrix = np.random.uniform(low=-jitter, 
                                      high=jitter, 
                                      size=(predictions_sensitve.shape[0] + 
                                            predictions_nonsensitve.shape[0],2))
    
    # ECDF-ified vals
    f_preds_nonsensitive = ecdf_dict['p_non_sensitive'](predictions_nonsensitve)
    f_preds_sensitive = ecdf_dict['p_sensitive'](predictions_sensitve)

    # non-sensitive return value
    vals_1 = np.zeros_like(predictions_nonsensitve)
    vals_1 += p_dict['p_non_sensitive']*eqf_dict['p_non_sensitive'](f_preds_nonsensitive)
    vals_1 += p_dict['p_sensitive']*eqf_dict['p_sensitive'](f_preds_nonsensitive)

    # sensitive return value
    vals_2 = np.zeros_like(predictions_sensitve)
    vals_2 += p_dict['p_non_sensitive']*eqf_dict['p_non_sensitive'](f_preds_sensitive)
    vals_2 += p_dict['p_sensitive']*eqf_dict['p_sensitive'](f_preds_sensitive)

    return vals_1, vals_2
