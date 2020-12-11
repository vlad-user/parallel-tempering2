import tensorflow as tf

import numpy as np
# import tensorflow_datasets as tfds
import deep_tempering as dt
from model_builders import lenet5_emnist_builder, lenet5_cifar10_builder, lenet5_cifar10_with_augmentation_builder
from utils import plot_error, log_exchange_data_mh, log_exchange_data_pbt, log_exchange_data_mh_temp_adj
from custom_callbacks import LogExchangeLossesCallback, MetropolisExchangeOneHPCallback, PBTExchangeTruncationSelectionCallback, MetropolisExchangeTempAdjustmentCallback
from keras.datasets import cifar10
from keras.utils import np_utils
from read_datasets import get_emnist_letters
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as shuffle_dataset

import wandb


wandb.init(
  project="deep-tempering",
  name="test-temp-swap-adj-log-differences-btw-betas-changed-clipping",
  notes="adjust temp by max half dist btw prev(or next) beta and old value for beta being adj",
  config={
    "model_name": "lenet5",
    "dataset_name": "cifar10",
    "model_builder": "lenet5_cifar10_builder",
    "hp_to_swap": 'learning_rate',
    "n_replicas": 8,
    "swap_step": 400,
    "burn_in": 25000,
    "batch_size": 128,
    "epochs": 400,
    "proba_coeff": 300,
    "train_data_size": 45000,
    "lr_range": [0.01, 0.015],
    "dropout_range": [0.4, 0.4],   # NOW DROPOUT PROBABILITY #ToDo: if value of hp==0 we get ZeroDivision error so if we need const dropout value != 0, we specify it
    "random_seed": 42,
    # "pbt_std": None,
    # "pbt_factor": [0.8, 1.2],
    "do_swap": True,
    'temp_adj_step': 10, #adj temp every 'temp adj step' exchange steps
}
)
#
config = wandb.config
import os
os.environ['PYTHONHASHSEED'] = str(config.random_seed)


import random
random.seed(config.random_seed)
np.random.seed(config.random_seed)
tf.set_random_seed(config.random_seed)
# from keras import backend as K
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)


model_builders = {"lenet5_cifar10_builder": lenet5_cifar10_builder, 'lenet5_emnist_builder': lenet5_emnist_builder,
                  "lenet5_cifar10_with_augmentation_builder": lenet5_cifar10_with_augmentation_builder}

hp = {1: {'learning_rate': [0.1 for _ in range(config.n_replicas)],
        'dropout_rate': np.linspace(config.dropout_range[0], config.dropout_range[1], config.n_replicas),},
       25000: {'learning_rate': np.linspace(config.lr_range[0], config.lr_range[1], config.n_replicas), #24993 - if epoch numbering starts from 1
        'dropout_rate': np.linspace(config.dropout_range[0], config.dropout_range[1], config.n_replicas),}
       }

hparams_dist_dict = {
  # 'learning_rate': lambda *x: np.random.normal(0.0, config.pbt_std),
    'learning_rate': lambda *x: np.random.choice([0.8, 1.2]),
  'dropout_rate': lambda *x: 0
  }


def prepare_data(config):
    if config.dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    elif config.dataset_name == 'emnist':
        x_train, y_train, x_test, y_test = get_emnist_letters()

        x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)),
           mode='constant', constant_values=0)
        x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)),
                         mode='constant', constant_values=0)
        y_train = np.int32(y_train) - 1
        y_test = np.int32(y_test) - 1


    x_train = np.float32(x_train) / 255.
    x_test = np.float32(x_test) / 255.


    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    x_train, y_train = shuffle_dataset(x_train, y_train, random_state=config.random_seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=config.random_seed)

    return x_train, y_train, x_val, y_val, x_test, y_test,


x_train, y_train, x_val, y_val, x_test, y_test = prepare_data(config)

assert x_train.shape[0] == config.train_data_size

model = dt.EnsembleModel(model_builders[config.model_builder])
model_noswap = dt.EnsembleModel(model_builders[config.model_builder])


model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              n_replicas=config.n_replicas)

model.summary()


history = model.fit(x_train,
                    y_train,
                    validation_data=(x_test, y_test),
                    hyper_params=hp,
                    batch_size=config.batch_size,
                    epochs=config.epochs,
                    # random_data_split_state=config.random_seed,
                    shuffle=True,
                    # callbacks=[PBTExchangeTruncationSelectionCallback(
                    #             exchange_data=(x_val,y_val),
                    #              swap_step=config.swap_step,
                    #              explore_weights=False,
                    #              explore_hyperparams=True,
                    #              burn_in=config.burn_in,
                    #              hyperparams_dist=hparams_dist_dict)]
                    callbacks=[MetropolisExchangeTempAdjustmentCallback(
                                                 exchange_data=(x_val,y_val),
                                                 swap_step=config.swap_step,
                                                 burn_in=config.burn_in,
                                                 coeff=config.proba_coeff,
                                                 hp_to_swap=config.hp_to_swap,
                                                 temp_adj_step=config.temp_adj_step)
                                ]
                    # callbacks=[LogExchangeLossesCallback(exchange_data=(x_val, y_val),
                    #                                      swap_step=config.swap_step,
                    #                                      burn_in=config.burn_in,)]
                    )




ex_history = history.exchange_history
history = history.history


for step in range(len(history['acc_0'])):
    for k in sorted(history.keys()):
    #     if k.endswith('0'):
    #         wandb.log({k.replace('_0', ''): history[k][step], 'epoch': step})
    #     else:
         wandb.log({k: history[k][step], 'epoch': step})

val_acc = np.array([history[f'val_acc_{i}'] for i in range(config.n_replicas)])
wandb.log({'best val acc, # of replica, step': [np.round(np.max(val_acc), 3), np.argmax(np.max(val_acc, axis=1)),
                                                    np.argmax(val_acc) % val_acc.shape[1]]})


log_exchange_data_mh_temp_adj(ex_history, config, config.do_swap)
# log_exchange_data_mh(ex_history, config, config.do_swap)

# log_exchange_data_pbt(ex_history, config)



# model_noswap.compile(optimizer=tf.keras.optimizers.SGD(),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'],
#               n_replicas=config.n_replicas)
#
#
# history_noswap = model_noswap.fit(x_train,
#                     y_train,
#                     validation_data=(x_test, y_test),
#                     hyper_params=hp,
#                     batch_size=config.batch_size,
#                     epochs=config.epochs,
#                    callbacks=[dt.callbacks.PBTExchangeCallback(
#                       exchange_data=(x_val,y_val),
#                       swap_step=None,
#                       explore_weights=False,
#                       explore_hyperparams=True,
#                       burn_in=0,
#                       hyperparams_dist=hparams_dist_dict)]
#                   # callbacks=[MetropolisExchangeOneHPCallback(
#                   #     exchange_data=(x_val, y_val),
#                   #     swap_step=None,
#                   #     burn_in=1,
#                   #     coeff=config.proba_coeff,
#                   #     hp_to_swap=config.hp_to_swap)]
#                                   )
#
# history_noswap = history_noswap.history
#
# val_acc_noswap = np.array([history_noswap[f'val_acc_{i}'] for i in range(config.n_replicas)])
# wandb.log({'best val acc no swap, # of replica, step': [np.round(np.max(val_acc_noswap), 3), np.argmax(np.max(val_acc_noswap, axis=1)), np.argmax(val_acc_noswap) % val_acc_noswap.shape[1]]})
# train_plt, test_plt = plot_error(config.n_replicas, config.batch_size, config.train_data_size, hp[config.burn_in][config.hp_to_exchange], history, history_noswap)
#
# wandb.log({"train error plot": wandb.Image(train_plt)})
# wandb.log({"test error plot": wandb.Image(test_plt)})
# wandb.log({"train error plot(i)": train_plt})
# wandb.log({"test error plot(i)": test_plt})
# # #
# # #
