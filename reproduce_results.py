import tensorflow as tf

import numpy as np
# import tensorflow_datasets as tfds
import deep_tempering as dt
from model_builders import lenet5_emnist_builder, lenet5_cifar10_builder, lenet5_cifar10_with_augmentation_builder, lenet5_cifar10_same_init_builder
from utils import *
from custom_callbacks import *
from keras.datasets import cifar10
from keras.utils import np_utils
from read_datasets import get_emnist_letters
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as shuffle_dataset

import sys
import os

import wandb

rs = sys.argv[1]





lr_range = int(sys.argv[2])

lr_ranges = [[0.007, 0.02], [0.01, 0.02], [0.02, 0.03], [0.04, 0.05]]


def pf(shape, min_val):
    return np.random.normal(0, 0, shape)

config={
    "model_name": "lenet5",
    "dataset_name": "cifar10",
    "model_builder": "lenet5_cifar10_same_init_builder",
    "hp_to_swap": 'learning_rate',
    "n_replicas": 8,
    "swap_step": 400,
    "burn_in": 25000,
    "batch_size": 128,
    "epochs": 500,
    "proba_coeff": 3,

    "train_data_size": 45000,
    "lr_range": lr_ranges[lr_range],
    "dropout_range": [0.4, 0.4],   # NOW DROPOUT PROBABILITY #ToDo: if value of hp==0 we get ZeroDivision error so if we need const dropout value != 0, we specify it
    "random_seed": int(rs),
    # "pbt_std": None,
    # "pbt_factor": [0.8, 1.2],
    "do_swap": True,
    'temp_adj_step': 10,#adj temp every 'temp adj step' exchange steps
    'n_prev_eval_steps': 8,
    'num_rear': 1,
    'rearrange_until': 25000,
    'burn_in_for_rearranger': 21200,
    'rearranger_perturb_std': 10,

}

# config["temp_sort_step"] = int((config['epochs'] * np.ceil(config['train_data_size'] / config['batch_size']) - config['burn_in'])) // 4
# print(config["temp_sort_step"])

wandb.init(
  project="deep-tempering",
  name=f"test-swap-sort-adj-diff-lr-ranges-{rs}",
  config=config,
  notes="",
)

config = wandb.config

import os
os.environ['PYTHONHASHSEED'] = str(config.random_seed)


import random
random.seed(config.random_seed)
np.random.seed(config.random_seed)
tf.set_random_seed(config.random_seed)


run_id = wandb.run.id

model_builders = {"lenet5_cifar10_builder": lenet5_cifar10_builder, 'lenet5_emnist_builder': lenet5_emnist_builder,
                  "lenet5_cifar10_with_augmentation_builder": lenet5_cifar10_with_augmentation_builder,
                  'lenet5_cifar10_same_init_builder': lenet5_cifar10_same_init_builder}

hp = {1: {'learning_rate': [0.1 for _ in range(config.n_replicas)],
        'dropout_rate': np.linspace(config.dropout_range[0], config.dropout_range[1], config.n_replicas),},
       20000: {'learning_rate': np.linspace(config.lr_range[0], config.lr_range[1], config.n_replicas), #24993 - if epoch numbering starts from 1
        'dropout_rate': np.linspace(config.dropout_range[0], config.dropout_range[1], config.n_replicas),}
       }

# hparams_dist_dict = {
#   # 'learning_rate': lambda *x: np.random.normal(0.0, config.pbt_std),
#     'learning_rate': lambda *x: np.random.choice([0.8, 1.2]),
#   'dropout_rate': lambda *x: 0
#   }





x_train, y_train, x_val, y_val, x_test, y_test = prepare_data(config)

assert x_train.shape[0] == config.train_data_size

model = dt.EnsembleModel(model_builders[config.model_builder])

model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              n_replicas=config.n_replicas)

model.summary()

weights_sort_clbk = WeightsSortCallback(exchange_data=(x_val, y_val),
                                       hp_to_swap=config.hp_to_swap,
                                       swap_step=400,
                                       burn_in=config.burn_in_for_rearranger,
                                       n_prev_eval_steps=config.n_prev_eval_steps,
                                       )

all_losses_clbk = LogExchangeLossesCallback(exchange_data=(x_val, y_val),
                                            hp_to_swap=config.hp_to_swap,
                                            swap_step=400,
                                            burn_in=0,
                                            n_prev_eval_steps=config.n_prev_eval_steps,
                                            do_swap=config.do_swap,
                                            weights_rear_clbk=weights_sort_clbk
                                            )
# rearanger_clbk = ReplicaRearrangerCallback(exchange_data=(x_val, y_val),
#                                            swap_step=400,
#                                            burn_in=config.burn_in_for_rearranger,
#                                            n_prev_eval_steps=config.n_prev_eval_steps,
#                                            perturb_func=None,
#                                            rear_step=(config.rearrange_until - config.burn_in) // 500 // (config.num_rear - 1) - 1 if config.num_rear > 1 else 100000,
#                                            rearrange_until=config.rearrange_until
#                                            )
# temp_sort_clbk = TempSortCallback(exchange_data=(x_val, y_val),
#                                   swap_step=400,
#                                   burn_in=config.burn_in_for_rearranger,
#                                   n_prev_eval_steps=config.n_prev_eval_steps,
#                                   n_replicas=config.n_replicas,
#                                   hp_to_swap=config.hp_to_swap
#                                   )




exch_clbk = MetropolisExchangeTempAdjustmentCallbackLogAllProbas(
                                                all_losses_clbk=all_losses_clbk,
                                                exchange_data=(x_val, y_val),
                                                hp_to_swap=config.hp_to_swap,
                                                swap_step=config.swap_step,
                                                burn_in=config.burn_in,
                                                coeff=config.proba_coeff,
                                                temp_adj_step=config.temp_adj_step,
                                                n_prev_eval_steps=config.n_prev_eval_steps,
                                                weights_sort_clbk=weights_sort_clbk

)

# exch_clbk = MetropolisExchangeOneHPCallbackLogAllProbas(
#                       exchange_data=(x_val, y_val),
#                       hp_to_swap=config.hp_to_swap,
#                       swap_step=config.swap_step,
#                       burn_in=config.burn_in,
#                       coeff=config.proba_coeff,
#                       n_prev_eval_steps=config.n_prev_eval_steps,
#                       weights_sort_clbk=weights_sort_clbk
#                       )

# exch_clbk = MetropolisExchangeTempSortCallbackLogAllProbas(
#                       temp_sort_clbk=temp_sort_clbk,
#                       exchange_data=(x_val, y_val),
#                       hp_to_swap=config.hp_to_swap,
#                       swap_step=config.swap_step,
#                       burn_in=config.burn_in,
#                       coeff=config.proba_coeff,
#                       )
# callbacks=[PBTExchangeTruncationSelectionCallback(
#             exchange_data=(x_val,y_val),
#              swap_step=config.swap_step,
#              explore_weights=False,
#              explore_hyperparams=True,
#              burn_in=config.burn_in,
#              hyperparams_dist=hparams_dist_dict)]

history = model.fit(x_train,
                    y_train,
                    validation_data=(x_test, y_test),
                    hyper_params=hp,
                    batch_size=config.batch_size,
                    epochs=config.epochs,
                    # random_data_split_state=config.random_seed,
                    shuffle=True,
                    callbacks=[all_losses_clbk, weights_sort_clbk, exch_clbk]
                   )




ex_history = exch_clbk.exchange_logs
addt_losses = all_losses_clbk.exchange_logs
# rear_history = rearanger_clbk.exchange_logs
history = history.history

for step in range(len(history['acc_0'])):
    for k in sorted(history.keys()):
         wandb.log({k: history[k][step], 'epoch': step})

val_acc = np.array([history[f'val_acc_{i}'] for i in range(config.n_replicas)])
wandb.log({'best val acc, # of replica, step': [np.round(np.max(val_acc), 3), np.argmax(np.max(val_acc, axis=1)),
                                                    np.argmax(val_acc) % val_acc.shape[1]]})
avg_num_temp_repl_visited, frac_visited_all_temp = calc_num_temp_replicas_visited(ex_history, config.n_replicas, replica_order=weights_sort_clbk.replica_order)
wandb.log({'avg_num_temp_repl_visited': avg_num_temp_repl_visited})
wandb.log({'frac_of_repl_visited_all_temp': frac_visited_all_temp})





log_exchange_data_mh_temp_adj_log_all_probas(ex_history, config, config.do_swap)

# log_exchange_data_mh_temp_sort_log_all_probas(ex_history, config, config.do_swap)

# log_exchange_data_mh_log_all_probas(ex_history, config, config.do_swap)

log_additional_losses(addt_losses, config, config.do_swap)



# log_exchange_data_pbt(ex_history, config)
