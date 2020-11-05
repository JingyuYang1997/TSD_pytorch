"""
CODE ADAPTED FROM: https://github.com/sjblim/rmsn_nips_2018

Implementation of Recurrent Marginal Structural Networks (R-MSNs):
Brian Lim, Ahmed M Alaa, Mihaela van der Schaar, "Forecasting Treatment Responses Over Time Using Recurrent
Marginal Structural Networks", Advances in Neural Information Processing Systems, 2018.
"""

import rmsn.configs

import tensorflow as tf
import numpy as np
import logging
import os
from tqdm import tqdm
import argparse

from rmsn.core_routines import train
import rmsn.core_routines as core

ROOT_FOLDER = rmsn.configs.ROOT_FOLDER
#MODEL_ROOT = configs.MODEL_ROOT
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


# EDIT ME! ################################################################################################
# Defines specific parameters to train for - skips hyperparamter optimisation if so
specifications = {
     'rnn_propensity_weighted': (0.1, 4, 100, 64, 0.01, 0.5),
     'treatment_rnn_action_inputs_only': (0.1, 3, 100, 128, 0.01, 2.0),
     'treatment_rnn': (0.1, 4, 100, 64, 0.01, 1.0),
}
####################################################################################################################


def rnn_fit(dataset_map, networks_to_train, MODEL_ROOT, b_use_predicted_confounders,
            b_use_oracle_confounders=False, b_remove_x1=False):

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    

    # Get the correct networks to train
    if networks_to_train == "propensity_networks":
        logging.info("Training propensity networks")
        net_names = ['treatment_rnn_action_inputs_only', 'treatment_rnn']

    elif networks_to_train == "encoder":
        logging.info("Training R-MSN encoder")
        net_names = ["rnn_propensity_weighted"]

    elif networks_to_train == "user_defined":
        logging.info("Training user defined network")
        raise NotImplementedError("Specify network to use!")

    else:
        raise ValueError("Unrecognised network type")

    logging.info("Running hyperparameter optimisation")

    # Experiment name
    expt_name = "treatment_effects"

    # Possible networks to use along with their activation functions
    activation_map = {'rnn_propensity_weighted': ("elu", 'linear'),
                      'rnn_propensity_weighted_logistic': ("elu", 'linear'),
                      'rnn_model': ("elu", 'linear'),
                      'treatment_rnn': ("tanh", 'sigmoid'),
                      'treatment_rnn_action_inputs_only': ("tanh", 'sigmoid')
                      }

    # Setup tensorflow
    tf_device = 'gpu'
    if tf_device == "cpu":
        config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 0})
    else:
        config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 1})
        config.gpu_options.allow_growth = True

    training_data = dataset_map['training_data']
    validation_data = dataset_map['validation_data']
    test_data = dataset_map['test_data']

    # Start Running hyperparam opt
    opt_params = {}
    for net_name in tqdm(net_names):

        # Re-run hyperparameter optimisation if parameters are not specified, otherwise train with defined params
        max_hyperparam_runs = 3 if net_name not in specifications else 1

        # Pull datasets
        b_predict_actions = "treatment_rnn" in net_name
        use_truncated_bptt = net_name != "rnn_model_bptt" # whether to train with truncated backpropagation through time
        b_propensity_weight = "rnn_propensity_weighted" in net_name
        b_use_actions_only = "rnn_action_inputs_only" in net_name


       # Extract only relevant trajs and shift data
        training_processed = core.get_processed_data(training_data, b_predict_actions,
                                                     b_use_actions_only, b_use_predicted_confounders,
                                                     b_use_oracle_confounders, b_remove_x1)
        validation_processed = core.get_processed_data(validation_data, b_predict_actions,
                                                       b_use_actions_only, b_use_predicted_confounders,
                                                       b_use_oracle_confounders, b_remove_x1)
        test_processed = core.get_processed_data(test_data, b_predict_actions,
                                                 b_use_actions_only, b_use_predicted_confounders,
                                                 b_use_oracle_confounders, b_remove_x1)

        num_features = training_processed['scaled_inputs'].shape[-1]
        num_outputs = training_processed['scaled_outputs'].shape[-1]

        # Load propensity weights if they exist
        if b_propensity_weight:

            if net_name == 'rnn_propensity_weighted_den_only':
                # use un-stabilised IPTWs generated by propensity networks
                propensity_weights = np.load(os.path.join(MODEL_ROOT, "propensity_scores_den_only.npy"))
            elif net_name == "rnn_propensity_weighted_logistic":
                # Use logistic regression weights
                propensity_weights = np.load(os.path.join(MODEL_ROOT, "propensity_scores.npy"))
                tmp = np.load(os.path.join(MODEL_ROOT, "propensity_scores_logistic.npy"))
                propensity_weights = tmp[:propensity_weights.shape[0], :, :]
            else:
                # use stabilised IPTWs generated by propensity networks
                propensity_weights = np.load(os.path.join(MODEL_ROOT, "propensity_scores.npy"))

            logging.info("Net name = {}. Mean-adjusting!".format(net_name))

            propensity_weights /= propensity_weights.mean()

            training_processed['propensity_weights'] = propensity_weights

        # Start hyperparamter optimisation
        hyperparam_count = 0
        while True:

            if net_name not in specifications:

                dropout_rate = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
                memory_multiplier = np.random.choice([0.5, 1, 2, 3, 4])
                num_epochs = 100
                minibatch_size = np.random.choice([64, 128, 256])
                learning_rate = np.random.choice([0.01, 0.005, 0.001])  #([0.01, 0.001, 0.0001])
                max_norm = np.random.choice([0.5, 1.0, 2.0, 4.0])
                hidden_activation, output_activation = activation_map[net_name]

            else:
                spec = specifications[net_name]
                logging.info("Using specifications for {}: {}".format(net_name, spec))
                dropout_rate = spec[0]
                memory_multiplier = spec[1]
                num_epochs = spec[2]
                minibatch_size = spec[3]
                learning_rate = spec[4]
                max_norm = spec[5]
                hidden_activation, output_activation = activation_map[net_name]

            model_folder = os.path.join(MODEL_ROOT, net_name)

            hyperparam_opt = train(net_name, expt_name,
                                  training_processed, validation_processed, test_processed,
                                  dropout_rate, memory_multiplier, num_epochs,
                                  minibatch_size, learning_rate, max_norm,
                                  use_truncated_bptt,
                                  num_features, num_outputs, model_folder,
                                  hidden_activation, output_activation,
                                  config,
                                  "hyperparam opt: {} of {}".format(hyperparam_count,
                                                                    max_hyperparam_runs))

            hyperparam_count = len(hyperparam_opt.columns)
            if hyperparam_count >= max_hyperparam_runs:
                opt_params[net_name] = hyperparam_opt.T
                break

        logging.info("Done")
        logging.info(hyperparam_opt.T)

        # Flag optimal params
    logging.info(opt_params)
