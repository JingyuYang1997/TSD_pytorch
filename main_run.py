import os
import argparse
import pickle
import logging

import numpy as np

from simulated_autoregressive import AutoregressiveSimulation
from time_series_deconfounder import time_series_deconfounder
from utils.evaluation_utils import load_results
from sklearn.model_selection import ShuffleSplit
from time_series_deconfounder import get_dataset_splits, train_rmsn

os.environ['CUDA_VISIBLE_DEVICES']='0'

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", default=0.6, type=float)
    parser.add_argument("--num_simulated_hidden_confounders", default=1, type=int)
    parser.add_argument("--num_substitute_hidden_confounders", default=1, type=int)
    parser.add_argument("--results_dir", default='results')
    parser.add_argument("--exp_name", default='test_tsd_gamma_0.6')
    parser.add_argument("--b_hyperparm_tuning", default=False)
    parser.add_argument("--train_and_get_confounder", action='store_true')
    parser.add_argument("--train_rmsn", action='store_false')
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    args = init_arg()

    model_name = 'factor_model'
    hyperparams_file = '{}/{}_best_hyperparams.txt'.format(args.results_dir, model_name)

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)


    dataset_with_confounders_filename = '{}/{}_dataset_with_substitute_confounders_tf.txt'.format(args.results_dir,
                                                                                               args.exp_name)
    factor_model_hyperparams_file = '{}/{}_factor_model_best_hyperparams.txt'.format(args.results_dir, args.exp_name)

    if args.train_and_get_confounder:
        np.random.seed(100)
        autoregressive = AutoregressiveSimulation(args.gamma, args.num_simulated_hidden_confounders)
        dataset = autoregressive.generate_dataset(5000, 31)
        time_series_deconfounder(dataset=dataset, num_substitute_confounders=args.num_substitute_hidden_confounders,
                                      exp_name=args.exp_name,
                                      dataset_with_confounders_filename=dataset_with_confounders_filename,
                                      factor_model_hyperparams_file=factor_model_hyperparams_file,
                                      b_hyperparm_tuning=args.b_hyperparm_tuning)

    if args.train_rmsn:
        dataset = load_results(dataset_with_confounders_filename)
        logging.info('Fitting counfounded recurrent marginal structural networks.')
        shuffle_split = ShuffleSplit(n_splits=1, test_size=0.1, random_state=10)
        train_index, test_index = next(shuffle_split.split(dataset['covariates'][:, :, 0]))
        shuffle_split = ShuffleSplit(n_splits=1, test_size=0.11, random_state=10)
        train_index, val_index = next(shuffle_split.split(dataset['covariates'][train_index, :, 0]))
        dataset_map = get_dataset_splits(dataset, train_index, val_index, test_index, use_predicted_confounders=True)

        logging.info('Fitting counfounded recurrent marginal structural networks.')
        rmse_without_confounders = train_rmsn(dataset_map, 'rmsn_' + str(args.exp_name), b_use_predicted_confounders=False)

        # logging.info(
        #     'Fitting deconfounded (D_Z = {}) recurrent marginal structural networks.'.format(
        #         args.num_substitute_confounders))
        rmse_with_confounders = train_rmsn(dataset_map, 'rmsn_' + str(args.exp_name), b_use_predicted_confounders=True)

        print("Outcome model RMSE when trained WITHOUT the hidden confounders.")
        print(rmse_without_confounders)

        print("Outcome model RMSE when trained WITH the substitutes for the hidden confounders.")
        print(rmse_with_confounders)
        print('done')