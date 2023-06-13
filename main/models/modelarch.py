"""
Module for Models used in the simulations

It contains a yoto-style network and vanilla networks
to compare to the baselines

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplingNetwork(nn.Module):
    """
    Sampler that produces the conditioning numbers
    """
    def __init__(self, 
                 input_dim, 
                 num_units, 
                 num_outputs, 
                 activation='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.num_units = num_units
        self.num_outputs = num_outputs
        self.activation = activation
        self.outlayers = nn.ModuleList()

        self.hidden_layer = nn.Linear(input_dim, num_units)
        for _ in range(num_outputs):
            self.outlayers.append(nn.Linear(num_units, 1))

    def forward(self, x):
        if self.activation == 'relu':
            x = F.relu(self.hidden_layer(x))
        else:
            x = self.hidden_layer(x)
        
        conditioning_values = []
        for out_ in self.outlayers:
            conditioning_values.append(out_(x))

        return conditioning_values
        

class ConditionedLayer(nn.Module):
    def __init__(self, 
                 input_dim,
                 num_units,
                 activation='relu', 
                 dropout=0.1):
        self.input_dim = input_dim
        self.num_units = num_units
        self.activation = activation
        self.dropout = dropout
        super().__init__()
        self._set_architecture()

    def _set_architecture(self):
        current_dim = self.input_dim
        self.hidden_layer = nn.Linear(current_dim, self.num_units)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x, mu, sigma):
        if self.activation == 'relu':
            x = F.relu(self.hidden_layer(x))
            if self.dropout is not None:
                x = self.dropout_layer(x)
        else:
            x = self.hidden_layer(x)
            if self.dropout is not None:
                x = self.dropout_layer(x)
        
        # Perform the yoto conditioning
        x = x*sigma + mu

        return x

class SampledMultitask(nn.Module):
    def __init__(self,
                 input_dim,
                 output_1,
                 sampling_out_1,
                 output_2,
                 sampling_out_2,
                 hidden_layer_architecture,
                 output_1_type='regression', 
                 output_2_type='regression',
                 hidden_units_conditioner=10,
                 device = torch.device('mps'), 
                 dropout=0.1, 
                 conditioning=True):
        super().__init__()
        self._check_architecture(hidden_layer_architecture, 
                                 sampling_out_1, 
                                 sampling_out_2)
        self.output_dim_1 = output_1
        self.output_dim_2 = output_2
        self.input_dim = input_dim
        self.device = device
        self.type_1 = output_1_type
        self.type_2 = output_2_type
        self.layers = nn.ModuleList()
        self.dropout = dropout
        self._set_architecture()

        # Set sampler as well
        if conditioning:
            self.sigma_network = SamplingNetwork(2,
                                                 hidden_units_conditioner, 
                                                 len(hidden_layer_architecture))
            self.mu_network = SamplingNetwork(2,
                                            hidden_units_conditioner, 
                                            len(hidden_layer_architecture))

    def _set_architecture(self):
        current_dim = self.input_dim

        if isinstance(self.architecture, int):
            self.layers.append(ConditionedLayer(current_dim, self.architecture, dropout=self.dropout))
            current_dim = self.architecture
        else:
            for layer_ in self.architecture:
                self.layers.append(ConditionedLayer(current_dim, layer_, dropout=self.dropout))
                current_dim = layer_
        
        self.output_1 = nn.Linear(current_dim, self.output_dim_1)
        self.output_2 = nn.Linear(current_dim, self.output_dim_2)
    
    def forward(self, x, param_=None):
        # If no parameter is passed, 
        # sample from distribution to get the weight
        if param_ is None:
            if self.training:
                param_ = self.get_sample(x.size()[0], 
                                         2, 
                                         self.sampling_params)
                # Add checks later

        # pass through the network layers
        # Need as a tensor for the auxiliary networks
        param_tensor = torch.cat(param_, 1).to(self.device)
        
        sigma = self.sigma_network(param_tensor)
        mu = self.mu_network(param_tensor)

        # calculate together
        for i, layer in enumerate(self.layers):
            x = layer(x, sigma[i], mu[i])
        
        # pass into output layers
        out1 = self.output_1(x)
        out2 = self.output_2(x)

        # do not need line below as we just use BCE logitloss
        # if self.type_2 == 'classification':
        #     out2 = torch.sigmoid(out2)
        
        if self.training:
            return out1, out2, param_
        else:
            return out1, out2
    
        
    def _check_architecture(self, architecture, sampling_1, sampling_2):
        assert isinstance(architecture, (list, int)),  "passed architecture should be list of int or int"
        if isinstance(architecture, list):
            assert all(isinstance(e_, int) for e_ in architecture), "passed architecture should be list of int or int"
        self.sampling_params = (sampling_1, sampling_2)
        self.architecture = architecture

    def get_sample(self, batch_size, num_losses, params,
                   sampling_dist='uniform'):
    # ToDo implement check for params and num losses
    # will expect a tuple of tuples of size num_losses
        if sampling_dist == 'uniform':
            weight_samples = []
            for loss_ in range(num_losses):
                loss_weights = ((params[loss_][0] - params[loss_][1]) *
                                torch.rand(batch_size, 1) +
                                params[loss_][1])
                
                #loss_weights.to(self.device)
                weight_samples.append(loss_weights)

        return weight_samples
    

class VanillaSingletask(nn.Module):
    def __init__(self, 
                 input_dim,
                 output_1,
                 hidden_layer_architecture,
                 output_1_type='regression', 
                 activation='relu',
                 device = torch.device('cuda'), 
                 dropout=0.1):
        super().__init__()
        self.architecture = hidden_layer_architecture
        self.output_dim_1 = output_1
        self.input_dim = input_dim
        self.device = device
        self.activation = activation
        self.type_1 = output_1_type
        self.layers = nn.ModuleList()
        self.dropout = dropout
        self._set_architecture()

    def _set_architecture(self):
        current_dim = self.input_dim

        if isinstance(self.architecture, int):
            self.layers.append(nn.Linear(current_dim, self.architecture))
            self.layers.append(nn.Dropout(self.dropout))
            current_dim = self.architecture
        else:
            for layer_ in self.architecture:
                self.layers.append(nn.Linear(current_dim, layer_))
                self.layers.append(nn.Dropout(self.dropout))
                current_dim = layer_
        
        self.output_1 = nn.Linear(current_dim, self.output_dim_1)

    def forward(self, x):
        if self.activation == 'relu':
            for lay_ in self.layers:
                if isinstance(lay_, torch.nn.modules.linear.Linear):
                    x = F.relu(lay_(x))
                else:
                    x = lay_(x)

        x = self.output_1(x)

        return x
    