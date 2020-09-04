import tensorflow as tf

# tf.enable_eager_execution()
import numpy as np
# import tensorflow_datasets as tfds
import deep_tempering as dt
from model_builders import lenet5_emnist_builder, lenet5_cifar10_builder
from utils import plot_error
from keras.datasets import cifar10
from keras.utils import np_utils
from read_datasets import get_emnist_letters
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as shuffle_dataset

import wandb
from wandb.keras import WandbCallback



wandb.init(
  project="deep-tempering",
  name="test-cifar10-lr-plot-swap-and-no-swap",
  notes="",
  config={
    "model_name": "lenet5",
    "dataset_name": "cifar10",
    "n_replicas": 8,
    "swap_step": 800,
    "burn_in": 25000,
    "batch_size": 128,
    "epochs": 400,
    "train_data_size": 45000,
    "lr_range": [0.015, 0.01],
    # "dropout_range": [0., 0.]  #ToDo: if value of hp==0 we get ZeroDivision error
}
)
config = wandb.config


model_builders = {'lenet5': {'cifar10': lenet5_cifar10_builder, 'emnist': lenet5_emnist_builder}}


def prepare_data(ds):
    if ds == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    elif ds == 'emnist':
        x_train, y_train, x_test, y_test = get_emnist_letters()

        x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), #ToDo: add augmentations
           mode='constant', constant_values=0)
        x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)),
                         mode='constant', constant_values=0)
        y_train = np.int32(y_train) - 1
        y_test = np.int32(y_test) - 1


    x_train = np.float32(x_train) / 255.
    x_test = np.float32(x_test) / 255.


    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    x_train, y_train = shuffle_dataset(x_train, y_train)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

    return x_train, y_train, x_val, y_val, x_test, y_test,


x_train, y_train, x_val, y_val, x_test, y_test = prepare_data(config.dataset_name)

assert x_train.shape[0] == config.train_data_size

model = dt.EnsembleModel(model_builders[config.model_name][config.dataset_name])
model_noswap = dt.EnsembleModel(model_builders[config.model_name][config.dataset_name])

hp1 = {'learning_rate': [0.01 for _ in range(config.n_replicas)],
       # 'dropout_rate': [0. for _ in range(config.n_replicas)]
       }

hp2 = {
    'learning_rate': np.linspace(config.lr_range[0], config.lr_range[1], config.n_replicas),
    # 'dropout_rate': np.linspace(config.dropout_range[0], config.dropout_range[1], config.n_replicas)
}

model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              n_replicas=config.n_replicas)


history1 = model.fit(x_train,
                    y_train,
                    validation_data=(x_val, y_val),
                    hyper_params=hp1,
                    batch_size=config.batch_size,
                    epochs=71, #71 epochs ~ 25K burn-in steps
                    swap_step=config.swap_step,
                    burn_in=config.burn_in,)
                    # callbacks=[WandbCallback(data_type="image",)]) wandbcallback doesnt work with EnsembleModel, it lacks several attributes



history2 = model.fit(x_train,
                    y_train,
                    validation_data=(x_val, y_val),
                    hyper_params=hp2,
                    batch_size=config.batch_size,
                    epochs=config.epochs - 71,
                    swap_step=config.swap_step,
                    burn_in=0,)

merged_history = {}
for k in history1.history:
    merged_history[k] = history1.history[k]
    merged_history[k].extend(history2.history[k])

for step in range(len(merged_history['acc_0'])):
    wandb.log({k: merged_history[k][step] for k in sorted(merged_history.keys())}, step=step)


model_noswap.compile(optimizer=tf.keras.optimizers.SGD(),
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              n_replicas=config.n_replicas)

history_noswap = model_noswap.fit(x_train,
                    y_train,
                    validation_data=(x_val, y_val),
                    hyper_params=hp2,
                    batch_size=config.batch_size,
                    epochs=400,
                    swap_step=None,
                    burn_in=0,)





# access the optimal (not compiled) keras' model instance
optimal_model = model.optimal_model()

# inference only on the trained optimal model
predicted = optimal_model.predict(x_test)

train_plt, test_plt = plot_error(config.n_replicas, config.batch_size, config.train_data_size, hp2['learning_rate'], merged_history, history_noswap.history)

wandb.log({"train error plot": wandb.Image(train_plt)})
wandb.log({"test error plot": wandb.Image(test_plt)})
wandb.log({"train error plot(i)": train_plt})
wandb.log({"test error plot(i)": test_plt})


