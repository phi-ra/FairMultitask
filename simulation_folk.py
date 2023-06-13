"""
Module for results in the "folktables" benchmarks, if this module is run 
from the command line, it requires two arguments:

Device: The device that is used to construct the torch models
Multiplication Factor: The factor by which the classification task
                       is to be multiplied in the prediction task

There are ten metrics to be calculated:

# Performance
MSE Multitask
AUC Multitask
MSE Singletask

# Fairness
Fairness MTL Regression
Fairness MTL Classification
Fairness STL Regression

# Perfomance in fairness
MSE MTL Fair
AUC MTL Fair

# Fairness in fairness
Fairness MTL Regression Fair
Fairness MTL Classification Fair

"""
# model loading
import torch

# wrangling
import numpy as np
import pandas as pd
import sys

from sklearn import metrics

# custom
from main.models.datasamplers import prepare_sim_data
from main.calibration.fairness import get_fairness_objects, get_fair_estimation
from main.calibration.custommetrics import unfairness_re
from utils.folkhelpers import train_all_models

if __name__ == '__main__':
    print('Arguments are device ("cpu", "cuda", "mps") and the lambda weight of task 2 relative to task 1')

    device_ = sys.argv[1]
    mult_factor = sys.argv[2]

    feature_df = pd.read_csv('data/prepared/folktabs/all_features.csv')
    label_df = pd.read_csv('data/prepared/folktabs/all_labels.csv')

    all_mods = []
    all_data = []
    all_seeds = [42, 49, 201, 49472, 928, 1, 7,58, 222, 7505, 6085, 288,
                 1570, 2900, 3287, 1689, 5898, 4119, 666, 7693]
    for seed_ in  all_seeds:
        tmp_data = prepare_sim_data(feature_df, 
                                    label_df,
                                    split_seed=seed_,
                                    missing_data=np.array([0,0.25,0.5,0.75,0.95]),
                                    missing_data_seed=seed_)
        
        all_data.append(tmp_data)
        trained_models = train_all_models(tmp_data,
                                          50,
                                          suffix_trial=f'_trial{seed_}', 
                                          device_=device_)
        
    data_and_seeds = dict(zip(all_seeds, all_data))

    # iterate through models, data and calculate metrics
    # Performance
    all_mse_multi = []
    all_auc_multi = []
    all_mse_single = []

    # Fairness
    all_fair_multi = []
    all_fair_multi_class = []
    all_fair_single = []

    # Perfomance in fairness
    all_auc_multi_fair = []
    all_mse_multi_fair = []

    # Fairness in fairness
    all_fair_multi_fair = []
    all_fair_multi_class_fair = []

    for seed, data_dict_tmp in data_and_seeds.items():
        seed_mse_multi = []
        seed_auc_multi = []
        seed_mse_single = []

        seed_fair_multi = []
        seed_fair_multi_class = []
        seed_fair_single = []

        seed_auc_multi_fair = []
        seed_mse_multi_fair = []

        seed_fair_multi_fair = []
        seed_fair_multi_class_fair = []

        for missing_prop, all_data_dict_tmp in data_dict_tmp.items():
            try:
                model_multi = torch.load(f'./models/multi_folk{missing_prop}__trial{seed}.pt')
                model_reg = torch.load(f'./models/regression_folk{missing_prop}__trial{seed}.pt')
               
            except RuntimeError:
                model_multi = torch.load(f'./models/multi_folk{missing_prop}__trial{seed}.pt', 
                                         map_location=torch.device('cpu'))
                model_reg = torch.load(f'./models/regression_folk{missing_prop}__trial{seed}.pt', 
                                       map_location=torch.device('cpu'))

            # Test predictions on all
            data_pred_tmmp = torch.from_numpy(all_data_dict_tmp['test'][0]).to(model_multi.device)
            sensitive_feature = all_data_dict_tmp['test'][0][:,5]

            model_multi.eval()
            model_reg.eval()

            model_multi.to(model_multi.device)
            model_reg.to(model_multi.device)

            weight_vector_1 = torch.ones((data_pred_tmmp.shape[0],1))*1
            weight_vector_2 = torch.ones((data_pred_tmmp.shape[0],1))*mult_factor

            weight_vector_1.to(model_multi.device)
            weight_vector_2.to(model_multi.device)
            test_lambda = [weight_vector_1, weight_vector_2]

            with torch.no_grad():    
                preds_y1, preds_y2 = model_multi(data_pred_tmmp,
                                                 param_=test_lambda)
                preds_reg = model_reg(data_pred_tmmp)

            preds_y2 = torch.sigmoid(preds_y2)

            preds_y1 = preds_y1.cpu().detach().numpy()
            preds_y2 = preds_y2.cpu().detach().numpy()
            preds_reg = preds_reg.cpu().detach().numpy()

            ## construct fairness of multi-tasker
            # multi
            multi_nonsens_reg = preds_y1[sensitive_feature == 0.0]
            multi_sens_reg = preds_y1[sensitive_feature == 1.0]

            multi_nonsens_class = preds_y2[sensitive_feature == 0.0]
            multi_sens_class = preds_y2[sensitive_feature == 1.0]

            #single
            single_nonsens_reg = preds_reg[sensitive_feature == 0.0]
            single_sens_reg = preds_reg[sensitive_feature == 1.0]

            fair_multi = unfairness_re(multi_nonsens_reg.reshape(-1,), multi_sens_reg.reshape(-1,))
            fair_multi_class = unfairness_re(multi_nonsens_class.reshape(-1,), multi_sens_class.reshape(-1,))
            fair_single = unfairness_re(single_nonsens_reg.reshape(-1,), single_sens_reg.reshape(-1,))

            mse_multi = metrics.mean_squared_error(all_data_dict_tmp['test'][1], 
                                                   preds_y1)
            
            auc_multi = metrics.roc_auc_score(all_data_dict_tmp['test'][2], 
                                              preds_y2)
            
            mse_single = metrics.mean_squared_error(all_data_dict_tmp['test'][1], 
                                                    preds_reg)
            
            seed_mse_multi.append(mse_multi)
            seed_auc_multi.append(auc_multi)
            seed_mse_single.append(mse_single)

            seed_fair_multi.append(fair_multi)
            seed_fair_multi_class.append(fair_multi_class)
            seed_fair_single.append(fair_single)

            #
            #   Get fair predictions
            #   for both single task and multitask
            #
            data_pred_tmmp = torch.from_numpy(all_data_dict_tmp['train_all'][0]).to(model_multi.device)
            sensitive_column = all_data_dict_tmp['train_all'][:, 5]

            non_sensitive_tensor = torch.from_numpy(all_data_dict_tmp['train_all'][0][sensitive_column == 0.0, :])
            sensitive_tensor = torch.from_numpy(all_data_dict_tmp['train_all'][0][sensitive_column == 1.0, :])

            non_sensitive_tensor= non_sensitive_tensor.to(model_multi.device)
            sensitive_tensor= sensitive_tensor.to(model_multi.device)

            weight_vector_1 = torch.ones((non_sensitive_tensor.shape[0],1))*1
            weight_vector_2 = torch.ones((non_sensitive_tensor.shape[0],1))*mult_factor

            weight_vector_1= weight_vector_1.to(model_multi.device)
            weight_vector_2= weight_vector_2.to(model_multi.device)

            test_lambda = [weight_vector_1, weight_vector_2]
            
            predictions_nonsensitve, predictions_nonsensitve_class = model_multi(non_sensitive_tensor, test_lambda)
            predictions_nonsensitve_class = torch.sigmoid(predictions_nonsensitve_class)

            weight_vector_1 = torch.ones((sensitive_tensor.shape[0],1))*1
            weight_vector_2 = torch.ones((sensitive_tensor.shape[0],1))*mult_factor
            weight_vector_1.to(device_)
            weight_vector_2.to(device_)

            test_lambda = [weight_vector_1, weight_vector_2]
            predictions_sensitive, predictions_sensitive_class = model_multi(sensitive_tensor, test_lambda)
            predictions_sensitive_class = torch.sigmoid(predictions_sensitive_class)

            p, e, q, = get_fairness_objects(sensitive_column, 
                        predictions_nonsensitve.cpu().detach().numpy(), 
                        predictions_sensitive.cpu().detach().numpy())
            
            p_c, e_c, q_c, = get_fairness_objects(sensitive_column, 
                        predictions_nonsensitve_class.cpu().detach().numpy(), 
                        predictions_sensitive_class.cpu().detach().numpy())
            
            # Get test predictions
            sensitive_column = all_data_dict_tmp['test'][0][:, 5]

            non_sensitive_tensor = torch.from_numpy(all_data_dict_tmp['test'][0][sensitive_column == 0.0, :])
            sensitive_tensor = torch.from_numpy(all_data_dict_tmp['test'][0][sensitive_column == 1.0, :])

            non_sensitive_tensor= non_sensitive_tensor.to(model_multi.device)
            sensitive_tensor= sensitive_tensor.to(model_multi.device)

            weight_vector_1 = torch.ones((non_sensitive_tensor.shape[0],1))*1
            weight_vector_2 = torch.ones((non_sensitive_tensor.shape[0],1))*2.75
            weight_vector_1= weight_vector_1.to(model_multi.device)
            weight_vector_2= weight_vector_2.to(model_multi.device)

            test_lambda = [weight_vector_1, weight_vector_2]
            predictions_nonsensitve, predictions_nonsensitve_class = model_multi(non_sensitive_tensor, test_lambda)
            predictions_nonsensitve_class = torch.sigmoid(predictions_nonsensitve_class)

            weight_vector_1 = torch.ones((sensitive_tensor.shape[0],1))*1
            weight_vector_2 = torch.ones((sensitive_tensor.shape[0],1))*2.75
            weight_vector_1= weight_vector_1.to(model_multi.device)
            weight_vector_2= weight_vector_2.to(model_multi.device)

            test_lambda = [weight_vector_1, weight_vector_2]
            predictions_sensitive, predictions_sensitive_class = model_multi(sensitive_tensor, test_lambda)
            predictions_sensitive_class = torch.sigmoid(predictions_sensitive_class)

            nonsenv_fair_class, sens_fair_class = get_fair_estimation(p_c,e_c,q_c,
                                                                      predictions_nonsensitve_class.cpu().detach().numpy().reshape(-1,), 
                                                                      predictions_sensitive_class.cpu().detach().numpy().reshape(-1,))
            
            nonsenv_fair, sens_fair = get_fair_estimation(p,e,q,
                                                        predictions_nonsensitve.cpu().detach().numpy().reshape(-1,), 
                                                        predictions_sensitive.cpu().detach().numpy().reshape(-1,))
            
            idx_0 = np.arange(len(sensitive_column))[sensitive_column == 0.0]
            idx_1 = np.arange(len(sensitive_column))[sensitive_column == 1.0]

            init_vector = np.zeros(shape=(len(sensitive_column), ))
            init_vector[idx_0] = nonsenv_fair
            init_vector[idx_1] = sens_fair

            init_vector_class = np.zeros(shape=(len(sensitive_column), ))
            init_vector_class[idx_0] = nonsenv_fair_class
            init_vector_class[idx_1] = sens_fair_class

            fair_multi_fair = unfairness_re(nonsenv_fair.reshape(-1,), sens_fair.reshape(-1,))
            fair_multi_class_fair = unfairness_re(nonsenv_fair_class.reshape(-1,), sens_fair_class.reshape(-1,))

            auc_multi_fair = metrics.roc_auc_score(all_data_dict_tmp['test'][2],
                                                   init_vector_class)
            mse_multi_fair = metrics.mean_squared_error(all_data_dict_tmp['test'][1], 
                                                    init_vector)
            seed_auc_multi_fair.append(auc_multi_fair)
            seed_mse_multi_fair.append(mse_multi_fair)

            seed_fair_multi_fair.append(fair_multi_fair)
            seed_fair_multi_class_fair.append(fair_multi_class_fair)

        all_mse_multi.append(seed_mse_multi)
        all_auc_multi.append(seed_auc_multi)
        all_mse_single.append(seed_mse_single)

        # Fairness
        all_fair_multi.append(seed_fair_multi)
        all_fair_multi_class.append(seed_fair_multi_class)
        all_fair_single.append(seed_fair_single)

        # Perfomance in fairness
        all_auc_multi_fair.append(seed_auc_multi_fair)
        all_mse_multi_fair.append(seed_mse_multi_fair)

        # Fairness in fairness
        all_fair_multi_fair.append(seed_fair_multi_fair)
        all_fair_multi_class_fair.append(seed_fair_multi_class_fair)
