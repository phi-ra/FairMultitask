"""
Module with helper functions to folktables simulation
"""
# NN
import torch
import torch.nn as nn
import torch.optim as optim

# custom
from main.models.modelarch import SampledMultitask, VanillaSingletask
from main.calibration.custommetrics import weighted_mse_loss

def train_single_model(model,
                       epochs,
                       train_data,
                       optimizer=None,
                       verbose=False, 
                       task='multi'):
    """
    Task1: Regression (weighted)
    Task2: Classification
    """
    loss_per_iter = []
    loss_per_batch = []

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(),
                               lr=0.001)
    else:
        optimizer = optimizer

    for epoch in range(epochs):
        if verbose:
            print(epoch)

        model.train(True)
        running_loss = 0.0
        for i, (xval, y1, y2, weights) in enumerate(train_data):
            xval = xval.to(model.device)
            y1 = y1.to(model.device)
            y2 = y2.to(model.device)
            weights = weights.to(model.device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            if task=='multi':

                outputs1, outputs2, lambdas_ = model(xval)
                lambda0 = lambdas_[0].to(model.device)
                lambda1 = lambdas_[1].to(model.device)

                weights_task1 = torch.reshape(lambda1*weights, (-1,1))
                weights_task1.to(model.device)
                loss_fct_task2 = nn.BCEWithLogitsLoss(weight=lambda0)

                
                loss_1 = weighted_mse_loss(outputs1, y1.float(), weights_task1)
                loss_2 = loss_fct_task2(outputs2, y2)
                
                # Combine losses and get gradient
                loss = loss_1 + loss_2

            elif task=='regression':
                outputs_reg = model(xval)
                weights = weights.to(model.device)

                weights_regression = torch.reshape(weights, (-1,1))
                weights_regression = weights_regression.to(model.device)
                loss = weighted_mse_loss(outputs_reg, y1.float(), weights_regression)

            elif task=='classification':
                outputs_cl = model(xval)

                weights = weights.to(model.device)
                loss_fct = nn.BCEWithLogitsLoss(weight=weights)
                loss = loss_fct(outputs_cl, y2)
               
            loss.backward()
            optimizer.step()

            # Save loss to plot
            running_loss += loss.item()
            loss_per_iter.append(loss.item())

        loss_per_batch.append(running_loss / (i + 1))
        running_loss = 0.0

    model.train(False)

def train_all_models(data_dict, 
                     epochs, 
                     suffix_trial, 
                     device_):
    device_=device_
    trained_model_dict = {}
    for key, data_pair in data_dict.items():
        trained_model_dict[key] = {}


        train_loader = data_pair['train']

        multitask_model = SampledMultitask(input_dim=87, 
                                   output_1=1, 
                                   sampling_out_1=(0.25, 4.25),
                                   output_2=1,
                                   output_2_type='classification',
                                   sampling_out_2=(0.25, 15),
                                   hidden_layer_architecture=[128,256,64],
                                   device=device_)
        
        multitask_model.to(device_)
        
        train_single_model(multitask_model,
                        epochs,
                        train_loader, 
                        task='multi')
        
        torch.save(multitask_model, f'./models/multi_folk{str(key)}_{suffix_trial}.pt')

        trained_model_dict[key]['multi'] = multitask_model

        simple_model_reg = VanillaSingletask(input_dim=87,
                        output_1=1,
                        hidden_layer_architecture=[128,256,64],
                        output_1_type='regression', 
                        activation='relu',
                        device =device_,
                        dropout=0.1)
        
        simple_model_reg.to(device_)
        
        train_single_model(simple_model_reg,
                        epochs,
                        train_loader, 
                        task='regression')
        trained_model_dict[key]['regression'] = simple_model_reg

        torch.save(simple_model_reg, f'./models/regression_folk{str(key)}_{suffix_trial}.pt')

        simple_model_class = VanillaSingletask(input_dim=87,
                        output_1=1,
                        hidden_layer_architecture=[128,256,64],
                        output_1_type='classification', 
                        activation='relu',
                        device = device_, 
                        dropout=0.1)
        
        train_single_model(simple_model_class,
                        epochs,
                        train_loader, 
                        task='classification')
        
        torch.save(simple_model_class, f'./models/classification_folk{str(key)}_{suffix_trial}.pt')
        
        trained_model_dict[key]['class'] = simple_model_class

    return trained_model_dict