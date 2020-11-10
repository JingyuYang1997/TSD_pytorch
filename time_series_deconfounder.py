import torch.nn
import torch
import numpy as np
import os
import shutil
from sklearn.model_selection import ShuffleSplit
import logging
from utils.torch_utils import dataset_base, get_dataset_splits
from factor_model import FactorModel
from torch.utils.data import DataLoader
from utils.evaluation_utils import write_results_to_file
from rmsn.script_rnn_fit import rnn_fit
from rmsn.script_rnn_test import rnn_test
from rmsn.script_propensity_generation import propensity_generation

def train_factor_model(dataset_train, dataset_val, dataset, num_confounders, hyperparams_file,
                       b_hyperparameter_optimisation):
    _, length, num_covariates = dataset_train['covariates'].shape
    num_treatments = dataset_train['treatments'].shape[-1]
    device = 'cuda'
    params = {'num_treatments': num_treatments,
              'num_covariates': num_covariates,
              'num_confounders': num_confounders,
              'max_sequence_length': length,
              'num_epochs': 1000}

    best_hyperparams = {
        'rnn_hidden_units': 128,
        'fc_hidden_units': 128,
        'learning_rate': 0.001,
        'batch_size': 2048,
        'rnn_keep_prob': 0.8}

    trainset = dataset_base(dataset_train)
    valset = dataset_base(dataset_val)
    allset = dataset_base(dataset)
    trainloader = DataLoader(trainset, batch_size=best_hyperparams['batch_size'], shuffle=True)
    valloader = DataLoader(valset, batch_size=best_hyperparams['batch_size'], shuffle=False)
    allloader = DataLoader(allset, batch_size=best_hyperparams['batch_size'], shuffle=False)

    factor_model = FactorModel(params=params, hyperparams=best_hyperparams, device=device)
    factor_model.train_model(trainloader,valloader)
    predicted_confounders = factor_model.compute_hidden_confounders(allloader)
    return predicted_confounders

def time_series_deconfounder(dataset, num_substitute_confounders, exp_name, dataset_with_confounders_filename,
                                  factor_model_hyperparams_file, b_hyperparm_tuning=False):
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    shuffle_split = ShuffleSplit(n_splits=1, test_size=0.1, random_state=10)
    train_index, test_index = next(shuffle_split.split(dataset['covariates'][:, :, 0]))
    shuffle_split = ShuffleSplit(n_splits=1, test_size=0.11, random_state=10)
    train_index, val_index = next(shuffle_split.split(dataset['covariates'][train_index, :, 0]))
    dataset_map = get_dataset_splits(dataset, train_index, val_index, test_index, use_predicted_confounders=False)

    dataset_train = dataset_map['training_data']
    dataset_val = dataset_map['validation_data']

    logging.info("Fitting factor model")
    predicted_confounders = train_factor_model(dataset_train, dataset_val,
                                               dataset,
                                               num_confounders=num_substitute_confounders,
                                               b_hyperparameter_optimisation=b_hyperparm_tuning,
                                               hyperparams_file=factor_model_hyperparams_file)
    dataset['predicted_confounders'] = predicted_confounders
    write_results_to_file(dataset_with_confounders_filename,dataset)

def train_rmsn(dataset_map, model_name, b_use_predicted_confounders):
    model_name = model_name + '_use_confounders_' + str(b_use_predicted_confounders)
    MODEL_ROOT = os.path.join('results', model_name)

    if not os.path.exists(MODEL_ROOT):
        os.mkdir(MODEL_ROOT)
        print("Directory ", MODEL_ROOT, " Created ")
    else:
        # Need to delete previously saved model.
        shutil.rmtree(MODEL_ROOT)
        os.mkdir(MODEL_ROOT)
        print("Directory ", MODEL_ROOT, " Created ")

    print("fitting propensity_networks")
    rnn_fit(dataset_map=dataset_map, networks_to_train='propensity_networks', MODEL_ROOT=MODEL_ROOT,
            b_use_predicted_confounders=b_use_predicted_confounders)
    print("generating propensity scores")
    propensity_generation(dataset_map=dataset_map, MODEL_ROOT=MODEL_ROOT,
                          b_use_predicted_confounders=b_use_predicted_confounders)
    print("training ")
    rnn_fit(networks_to_train='encoder', dataset_map=dataset_map, MODEL_ROOT=MODEL_ROOT,
            b_use_predicted_confounders=b_use_predicted_confounders)

    rmsn_mse = rnn_test(dataset_map=dataset_map, MODEL_ROOT=MODEL_ROOT,
                        b_use_predicted_confounders=b_use_predicted_confounders)

    rmse = np.sqrt(np.mean(rmsn_mse)) * 100
    return rmse
