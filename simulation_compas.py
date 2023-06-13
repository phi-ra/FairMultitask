"""
Module for results in the "compas" dataset
"""
# NN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# wrangling
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# custom
from main.models.modelarch import SampledMultitask, VanillaSingletask
from main.models.datasamplers import TabularDataSet
from main.calibration.fairness import get_fairness_objects, get_fair_estimation
from main.calibration.custommetrics import unfairness_re

if __name__ == "__main__":
    # can also change to cuda or mps
    device_ = 'cpu'

    df_all = pd.read_csv('data/raw/compas/'+ 
                         'compas-scores-two-years.csv')
    
    # analogous to the original article, remove some observations
    df_want = df_all.copy()
    df_want = df_want.loc[(df_want.days_b_screening_arrest <= 30), :]
    df_want = df_want.loc[(df_want.days_b_screening_arrest >= -30), :]
    df_want = df_want.loc[df_want.is_recid != -1, :]
    df_want = df_want.loc[df_want.c_charge_degree != 'O', :]
    df_want = df_want.loc[~df_want.score_text.isna(), :]

    df_want['possession_dummy'] = df_want.c_charge_desc.apply(lambda x: 'poss' in str(x).lower())*1
    df_want['theft_dummy'] = df_want.c_charge_desc.apply(lambda x: 'theft' in str(x).lower())*1
    df_want['driv_dummy'] = df_want.c_charge_desc.apply(lambda x: 'driv' in str(x).lower())*1
    df_want['battery_dummy'] = df_want.c_charge_desc.apply(lambda x: 'batt' in str(x).lower())*1
    df_want['assault_dummy'] = df_want.c_charge_desc.apply(lambda x: 'assault' in str(x).lower())*1
    df_want['weapon_dummy'] = df_want.c_charge_desc.apply(lambda x: 'weap' in str(x).lower())*1
    df_want['firearm_dummy'] = df_want.c_charge_desc.apply(lambda x: 'arm' in str(x).lower())*1

    df_want.drop(columns='c_charge_desc', inplace=True)

    df_features = df_want.drop(columns=['is_violent_recid', 'is_recid'])
    df_labels = df_want.loc[:, ['is_recid', 'is_violent_recid']]   

    remove_features = list(df_want.isna().sum(axis=0)[df_want.isna().sum(axis=0) > 1].index)
    remove_features = remove_features + ['two_year_recid', 'event', 'id', 'name', 'first', 'last', 
                                            'compas_screening_date', 'dob', 'c_jail_in', 'c_jail_out', 
                                            'c_case_number', 'type_of_assessment', 'screening_date', 
                                            'v_type_of_assessment', 'v_screening_date', 'in_custody', 
                                            'out_custody', 
                                            'decile_score.1', 'v_decile_score', 'decile_score', 
                                            'score_text', 'v_score_text', 
                                            'priors_count.1', 'start', 'end']

    df_features = df_features.loc[:, [col for col in df_features.columns if not col in remove_features]]

    other_cols = list(df_features.select_dtypes('object').columns)
    numeric_cols = [col_ for col_ in df_features.columns if not col_ in other_cols]

    onehot_ = OneHotEncoder()
    scaler_ = StandardScaler()

    transformer = ColumnTransformer(
        transformers=[('categoricals', onehot_, other_cols), 
                    ('numerical', scaler_, numeric_cols)
                    ], 
                    remainder='passthrough', 
                    sparse_threshold=0)

    trial_dict = {}
    # seeds were generated using google 
    for seed_ in [42, 49, 201, 49472, 928, 1, 7,58, 222, 7505, 6085, 288,
                  1570, 2900, 3287, 1689, 5898, 4119, 666, 7693]:

        X_train, X_test, y_train, y_test = train_test_split(df_features,
                                                            df_labels, 
                                                            test_size=0.2,
                                                            random_state=seed_)

        transformer.fit(X_train)
        X_scaled = np.float32(transformer.transform(X_train))
        X_test_scaled = np.float32(transformer.transform(X_test))


        data_train = TabularDataSet(X_scaled,
                                    np.float32(y_train.iloc[:,0]).reshape(-1,1),
                                    np.float32(y_train.iloc[:,1]).reshape(-1,1))
        data_test = TabularDataSet(X_test_scaled,
                                np.float32(y_test.iloc[:,0]).reshape(-1,1),
                                np.float32(y_test.iloc[:,1]).reshape(-1,1))

        # Dataloaders
        trainloader = DataLoader(data_train, batch_size=128, shuffle=True)
        testloader = DataLoader(data_test, batch_size=128, shuffle=False)

        multitask_model = SampledMultitask(input_dim=27, 
                                        output_1=1, 
                                        sampling_out_1=(0.25, 6),
                                        output_2=1,
                                        sampling_out_2=(0.25, 6),
                                        hidden_layer_architecture=[24,24,24],
                                        device=device_)

        regression_model_1 = VanillaSingletask(input_dim=27,
                                output_1=1,
                                hidden_layer_architecture=[24,24,24],
                                output_1_type='classification', 
                                activation='relu',
                                device = device_, 
                                dropout=0.1)

        regression_model_2 = VanillaSingletask(input_dim=27,
                                output_1=1,
                                hidden_layer_architecture=[24,24,24],
                                output_1_type='classification', 
                                activation='relu',
                                device = device_, 
                                dropout=0.1)


        loss_per_iter = []
        loss_per_batch = []

        optimizer = optim.Adam(regression_model_1.parameters(),
                            lr=0.001)

        for epoch in range(50):
            #print(epoch)
            regression_model_1.train(True)
            running_loss = 0.0
            for i, (xval, y1, y2) in enumerate(trainloader):
                xval = xval.to(device_)
                y1 = y1.to(device_)
                y2 = y2.to(device_)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs1= regression_model_1(xval.float())

                # need to set loss weights resulting from zero-loss observations to zero        
                loss_fct_binary_task_1 = nn.BCEWithLogitsLoss()

                loss_1 = loss_fct_binary_task_1(outputs1, y1)
                
                # Combine losses and get gradient
                loss = loss_1 
                loss.backward()
                optimizer.step()

                # Save loss to plot
                running_loss += loss.item()
                loss_per_iter.append(loss.item())

            loss_per_batch.append(running_loss / (i + 1))
            running_loss = 0.0

        regression_model_1.train(False)

        optimizer = optim.Adam(regression_model_2.parameters(),
                            lr=0.001)

        for epoch in range(50):
            #print(epoch)
            regression_model_2.train(True)
            running_loss = 0.0
            for i, (xval, y1, y2) in enumerate(trainloader):
                xval = xval.to(device_)
                y1 = y1.to(device_)
                y2 = y2.to(device_)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs1= regression_model_2(xval.float())

                # need to set loss weights resulting from zero-loss observations to zero        
                loss_fct_binary_task_1 = nn.BCEWithLogitsLoss()

                loss_1 = loss_fct_binary_task_1(outputs1, y2)
                
                # Combine losses and get gradient
                loss = loss_1 
                loss.backward()
                optimizer.step()

                # Save loss to plot
                running_loss += loss.item()
                loss_per_iter.append(loss.item())

            loss_per_batch.append(running_loss / (i + 1))
            running_loss = 0.0

        regression_model_2.train(False)


        optimizer = optim.Adam(multitask_model.parameters(),
                            lr=0.001)

        for epoch in range(75):
            #print(epoch)
            multitask_model.train(True)
            running_loss = 0.0
            for i, (xval, y1, y2) in enumerate(trainloader):
                xval = xval.to(device_)
                y1 = y1.to(device_)
                y2 = y2.to(device_)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs1, outputs2, lambdas_ = multitask_model(xval.float())
                lambda0 = lambdas_[0].to(device_)
                lambda1 = lambdas_[1].to(device_)

                # need to set loss weights resulting from zero-loss observations to zero        
                loss_fct_binary_task_1 = nn.BCEWithLogitsLoss(weight=lambda0)
                loss_fct_binary_task_2 = nn.BCEWithLogitsLoss(weight=lambda1)

                loss_1 = loss_fct_binary_task_1(outputs1, y1)
                loss_2 = loss_fct_binary_task_2(outputs2, y2)
                
                # Combine losses and get gradient
                loss = loss_1 + loss_2
                loss.backward()
                optimizer.step()

                # Save loss to plot
                running_loss += loss.item()
                loss_per_iter.append(loss.item())

            loss_per_batch.append(running_loss / (i + 1))
            running_loss = 0.0

        multitask_model.train(False)


        # Evaluate
        weight_check_1 = 1
        weight_check_2 = 6

        sensitive_feature_ = X_scaled[:,5]
        all_test = X_test_scaled

        y_1_test = y_test.iloc[:,0].to_numpy().reshape(-1,1)
        y_2_test = y_test.iloc[:,1].to_numpy().reshape(-1,1)

        all_x_test = torch.from_numpy(all_test).to(device_)

        nonsensitive_vec = X_scaled[sensitive_feature_ == 0.0,:]
        sensitive_vec = X_scaled[sensitive_feature_ == 1.0,:]
        nonsensitive_vec = torch.from_numpy(nonsensitive_vec).to(device_)
        sensitive_vec = torch.from_numpy(sensitive_vec).to(device_)

        with torch.no_grad():
            weight_vector_1 = torch.ones((all_x_test.shape[0],1))*weight_check_1
            weight_vector_2 = torch.ones((all_x_test.shape[0],1))*weight_check_2
            weight_vector_1 = weight_vector_1.to(device_)
            weight_vector_2 = weight_vector_2.to(device_)

            test_lambda = [weight_vector_1, weight_vector_2]
            preds_y1, preds_y2 = multitask_model(all_x_test, param_=test_lambda)
            preds_y1 = torch.sigmoid(preds_y1).detach().numpy()
            preds_y2 = torch.sigmoid(preds_y2).detach().numpy()


            weight_vector_1 = torch.ones((nonsensitive_vec.shape[0],1))*weight_check_1
            weight_vector_2 = torch.ones((nonsensitive_vec.shape[0],1))*weight_check_2
            weight_vector_1 = weight_vector_1.to(device_)
            weight_vector_2 = weight_vector_2.to(device_)

            test_lambda = [weight_vector_1, weight_vector_2]
            preds_y1_ns, preds_y2_ns = multitask_model(nonsensitive_vec, param_=test_lambda)
            preds_y1_ns = torch.sigmoid(preds_y1_ns).detach().numpy()
            preds_y2_ns = torch.sigmoid(preds_y2_ns).detach().numpy()

            weight_vector_1 = torch.ones((sensitive_vec.shape[0],1))*weight_check_1
            weight_vector_2 = torch.ones((sensitive_vec.shape[0],1))*weight_check_2
            weight_vector_1 = weight_vector_1.to(device_)
            weight_vector_2 = weight_vector_2.to(device_)

            test_lambda = [weight_vector_1, weight_vector_2]
            preds_y1_s, preds_y2_s = multitask_model(sensitive_vec, param_=test_lambda)
            preds_y1_s = torch.sigmoid(preds_y1_s).detach().numpy()
            preds_y2_s = torch.sigmoid(preds_y2_s).detach().numpy()

            preds_y1_single = regression_model_1(all_x_test)
            preds_y2_single = regression_model_2(all_x_test)
            preds_y1_single = torch.sigmoid(preds_y1_single).detach().numpy()
            preds_y2_single = torch.sigmoid(preds_y2_single).detach().numpy()

            preds_y1_single_ns = regression_model_1(nonsensitive_vec)
            preds_y2_single_ns = regression_model_2(nonsensitive_vec)
            preds_y1_single_ns = torch.sigmoid(preds_y1_single_ns).detach().numpy()
            preds_y2_single_ns = torch.sigmoid(preds_y2_single_ns).detach().numpy()

            preds_y1_single_s = regression_model_1(sensitive_vec)
            preds_y2_single_s = regression_model_2(sensitive_vec)
            preds_y1_single_s = torch.sigmoid(preds_y1_single_s).detach().numpy()
            preds_y2_single_s = torch.sigmoid(preds_y2_single_s).detach().numpy()

        # calculate stats
        auc_1_unfair = metrics.roc_auc_score(y_1_test, preds_y1)
        auc_2_unfair = metrics.roc_auc_score(y_1_test, preds_y2)
        unfairness_1 = unfairness_re(preds_y1_ns.reshape(-1,), preds_y1_s.reshape(-1,))
        unfairness_2 = unfairness_re(preds_y2_ns.reshape(-1,), preds_y2_s.reshape(-1,))

        auc_1_unfair_single = metrics.roc_auc_score(y_1_test, preds_y1_single)
        auc_2_unfair_single = metrics.roc_auc_score(y_1_test, preds_y2_single)
        unfairness_1_single = unfairness_re(preds_y1_single_ns.reshape(-1,), preds_y1_single_s.reshape(-1,))
        unfairness_2_single = unfairness_re(preds_y2_single_ns.reshape(-1,), preds_y2_single_s.reshape(-1,))

        # Transform to fair
        p1, e1, q1, = get_fairness_objects(sensitive_feature_, 
                                            preds_y1_ns, 
                                            preds_y1_s)

        p2, e2, q2, = get_fairness_objects(sensitive_feature_, 
                                            preds_y2_ns, 
                                            preds_y2_s)

        # Transform to fair
        p1_sing, e1_sing, q1_sing, = get_fairness_objects(sensitive_feature_, 
                                            preds_y1_single_ns, 
                                            preds_y1_single_s)

        p2_sing, e2_sing, q2_sing, = get_fairness_objects(sensitive_feature_, 
                                            preds_y2_single_ns, 
                                            preds_y2_single_s)
        # Now get test preds 
        sensitive_feature_ = X_test_scaled[:,5]
        nonsensitive_vec = X_test_scaled[sensitive_feature_ == 0.0,:]
        sensitive_vec = X_test_scaled[sensitive_feature_ == 1.0,:]
        nonsensitive_vec = torch.from_numpy(nonsensitive_vec).to(device_)
        sensitive_vec = torch.from_numpy(sensitive_vec).to(device_)

        with torch.no_grad():
            weight_vector_1 = torch.ones((nonsensitive_vec.shape[0],1))*weight_check_1
            weight_vector_2 = torch.ones((nonsensitive_vec.shape[0],1))*weight_check_2
            weight_vector_1 = weight_vector_1.to(device_)
            weight_vector_2 = weight_vector_2.to(device_)

            test_lambda = [weight_vector_1, weight_vector_2]
            preds_y1_ns, preds_y2_ns = multitask_model(nonsensitive_vec, param_=test_lambda)
            preds_y1_ns = torch.sigmoid(preds_y1_ns).detach().numpy()
            preds_y2_ns = torch.sigmoid(preds_y2_ns).detach().numpy()

            weight_vector_1 = torch.ones((sensitive_vec.shape[0],1))*weight_check_1
            weight_vector_2 = torch.ones((sensitive_vec.shape[0],1))*weight_check_2
            weight_vector_1 = weight_vector_1.to(device_)
            weight_vector_2 = weight_vector_2.to(device_)

            test_lambda = [weight_vector_1, weight_vector_2]
            preds_y1_s, preds_y2_s = multitask_model(sensitive_vec, param_=test_lambda)
            preds_y1_s = torch.sigmoid(preds_y1_s).detach().numpy()
            preds_y2_s = torch.sigmoid(preds_y2_s).detach().numpy()

            preds_y1_single_ns = regression_model_1(nonsensitive_vec)
            preds_y2_single_ns = regression_model_2(nonsensitive_vec)
            preds_y1_single_ns = torch.sigmoid(preds_y1_single_ns).detach().numpy()
            preds_y2_single_ns = torch.sigmoid(preds_y2_single_ns).detach().numpy()

            preds_y1_single_s = regression_model_1(sensitive_vec)
            preds_y2_single_s = regression_model_2(sensitive_vec)
            preds_y1_single_s = torch.sigmoid(preds_y1_single_s).detach().numpy()
            preds_y2_single_s = torch.sigmoid(preds_y2_single_s).detach().numpy()


        nonsenv_fair1, sens_fair1 = get_fair_estimation(p1,e1,q1,
                                                        preds_y1_ns.reshape(-1,), 
                                                        preds_y1_s.reshape(-1,))



        nonsenv_fair2, sens_fair2 = get_fair_estimation(p2,e2,q2,
                                                        preds_y2_ns.reshape(-1,), 
                                                        preds_y2_s.reshape(-1,))

        nonsenv_fair1_single, sens_fair1_single = get_fair_estimation(p1_sing,e1_sing,q1_sing,
                                                        preds_y1_single_ns.reshape(-1,), 
                                                        preds_y1_single_s.reshape(-1,))

        nonsenv_fair2_single, sens_fair2_single = get_fair_estimation(p2_sing,e2_sing,q2_sing,
                                                        preds_y2_single_ns.reshape(-1,), 
                                                        preds_y2_single_s.reshape(-1,))


        idx_0 = np.arange(len(sensitive_feature_))[sensitive_feature_ == 0.0]
        idx_1 = np.arange(len(sensitive_feature_))[sensitive_feature_ == 1.0]

        init_vector = np.zeros(shape=(len(sensitive_feature_), ))
        init_vector[idx_0] = nonsenv_fair1
        init_vector[idx_1] = sens_fair1

        init_vector1 = np.zeros(shape=(len(sensitive_feature_), ))
        init_vector1[idx_0] = nonsenv_fair2
        init_vector1[idx_1] = sens_fair2

        init_vector_single = np.zeros(shape=(len(sensitive_feature_), ))
        init_vector_single[idx_0] = nonsenv_fair1_single
        init_vector_single[idx_1] = sens_fair1_single

        init_vector1_single = np.zeros(shape=(len(sensitive_feature_), ))
        init_vector1_single[idx_0] = nonsenv_fair2_single
        init_vector1_single[idx_1] = sens_fair2_single

        auc_1_fair = metrics.roc_auc_score(y_1_test, init_vector)
        auc_2_fair = metrics.roc_auc_score(y_2_test, init_vector1)

        unfairness_1_fair = unfairness_re(nonsenv_fair1.reshape(-1,), sens_fair1.reshape(-1,))
        unfairness_2_fair = unfairness_re(nonsenv_fair2.reshape(-1,), sens_fair2.reshape(-1,))

        auc_1_fair_single = metrics.roc_auc_score(y_1_test, init_vector_single)
        auc_2_fair_single = metrics.roc_auc_score(y_2_test, init_vector1_single)

        unfairness_1_fair_single = unfairness_re(nonsenv_fair1_single.reshape(-1,), sens_fair1_single.reshape(-1,))
        unfairness_2_fair_single = unfairness_re(nonsenv_fair2_single.reshape(-1,), sens_fair2_single.reshape(-1,))

        trial_dict[seed_] = {}
        trial_dict[seed_]['auc_1_unfair'] = auc_1_unfair
        trial_dict[seed_]['auc_2_unfair'] = auc_2_unfair
        trial_dict[seed_]['unfairness_1'] = unfairness_1
        trial_dict[seed_]['unfairness_2'] = unfairness_2
        trial_dict[seed_]['auc_1_unfair_single'] = auc_1_unfair_single
        trial_dict[seed_]['auc_2_unfair_single'] = auc_2_unfair_single
        trial_dict[seed_]['unfairness_1_single'] = unfairness_1_single
        trial_dict[seed_]['unfairness_2_single'] = unfairness_2_single
        trial_dict[seed_]['auc_1_fair'] = auc_1_fair
        trial_dict[seed_]['unfairness_1_fair'] = unfairness_1_fair
        trial_dict[seed_]['auc_2_fair'] = auc_2_fair
        trial_dict[seed_]['unfairness_2_fair'] = unfairness_2_fair
        trial_dict[seed_]['auc_1_fair_single'] = auc_1_fair_single
        trial_dict[seed_]['unfairness_1_fair_single'] = unfairness_1_fair_single
        trial_dict[seed_]['auc_2_fair_single'] = auc_2_fair_single
        trial_dict[seed_]['unfairness_2_fair_single'] = unfairness_2_fair_single
