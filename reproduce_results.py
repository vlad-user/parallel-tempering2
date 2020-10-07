import tensorflow as tf

# tf.enable_eager_execution()
import numpy as np
# import tensorflow_datasets as tfds
import deep_tempering as dt
from model_builders import lenet5_emnist_builder, lenet5_cifar10_builder, lenet5_cifar10_with_augmentation_builder
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
  name="test-cifar10-lr-log-exchange",
  notes="",
  config={
    "model_name": "lenet5",
    "dataset_name": "cifar10",
    "model_builder": "lenet5_cifar10_builder",
    "n_replicas": 2,
    "swap_step": 800,
    "burn_in": 25000,
    "batch_size": 128,
    "epochs": 400,
    "proba_coeff": 300000,
    "train_data_size": 45000,
    "lr_range": [0.015, 0.01],
    "dropout_range": [0.999, 0.999]  #ToDo: if value of hp==0 we get ZeroDivision error so if we need const dropout value != 0, we specify it
}
)

config = wandb.config


model_builders = {"lenet5_cifar10_builder": lenet5_cifar10_builder, 'lenet5_emnist_builder': lenet5_emnist_builder,
                  "lenet5_cifar10_with_augmentation_builder": lenet5_cifar10_with_augmentation_builder}

hp = {1: {'learning_rate': [0.1 for _ in range(config.n_replicas)],
        'dropout_rate': np.linspace(config.dropout_range[0], config.dropout_range[1], config.n_replicas),},
       71: {'learning_rate': np.linspace(config.lr_range[0], config.lr_range[1], config.n_replicas),
        'dropout_rate': np.linspace(config.dropout_range[0], config.dropout_range[1], config.n_replicas),}
       }


def prepare_data(ds):
    if ds == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    elif ds == 'emnist':
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

    x_train, y_train = shuffle_dataset(x_train, y_train)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

    return x_train, y_train, x_val, y_val, x_test, y_test,


x_train, y_train, x_val, y_val, x_test, y_test = prepare_data(config.dataset_name)

assert x_train.shape[0] == config.train_data_size

model = dt.EnsembleModel(model_builders[config.model_builder])
model_noswap = dt.EnsembleModel(model_builders[config.model_builder])


class SGDLearningRateTracker(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        lrs = [tf.keras.backend.eval(self.model._train_attrs[i]['optimizer'].lr * (1. / (1. + self.model._train_attrs[i]['optimizer'].decay *
                                                                                         tf.cast(self.model._train_attrs[i]['optimizer'].iterations, dtype=tf.float32)))) for i in range(len(self.model._train_attrs))]
        print(lrs)

model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              n_replicas=config.n_replicas)

# model.summary()


history = model.fit(x_train,
                    y_train,
                    validation_data=(x_val, y_val),
                    hyper_params=hp,
                    batch_size=config.batch_size,
                    epochs=config.epochs,
                    swap_step=config.swap_step,
                    burn_in=config.burn_in,
                    # coeff=config.proba_coeff
                    # callbacks=[WandbCallback(data_type="image",)]) #wandbcallback doesnt work with EnsembleModel, it lacks several attributes
                    # callbacks=[SGDLearningRateTracker(),
                    #            tf.keras.callbacks.TensorBoard(log_dir='./logs/test/', write_graph=False, update_freq='batch'),
                    #            ]
                    )




ex_history = history.exchange_history
history = history.history

print(ex_history)
for step in range(len(history['acc_0'])):
    wandb.log({k: history[k][step] for k in sorted(history.keys())}, step=step)

for i, step in enumerate(ex_history['step']):
    wandb.log({'exchange probas': ex_history['proba'][i]}, step=step)
    # wandb.log({'acceptance ratio': ex_history['accept_ratio'][i]}, step=step)

wandb.log({'num of exchange attempts': len(ex_history['proba'])})
wandb.log({'num of exchanges': ex_history['swaped'].count(1)})

val_acc = np.array([history[f'val_acc_{i}'] for i in range(config.n_replicas)])
wandb.log({'best val acc, # of replica, step': [np.round(np.max(val_acc), 3), np.argmax(np.max(val_acc, axis=1)), np.argmax(val_acc) % val_acc.shape[1]]})







model_noswap.compile(optimizer=tf.keras.optimizers.SGD(),
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              n_replicas=config.n_replicas)


history_noswap = model_noswap.fit(x_train,
                    y_train,
                    validation_data=(x_val, y_val),
                    hyper_params=hp,
                    batch_size=config.batch_size,
                    epochs=config.epochs,
                    swap_step=None,
                    burn_in=0,
                                    )

history_noswap = history_noswap.history


# access the optimal (not compiled) keras' model instance
optimal_model = model.optimal_model()

# inference only on the trained optimal model
predicted = optimal_model.predict(x_test)


train_plt, test_plt = plot_error(config.n_replicas, config.batch_size, config.train_data_size, hp['learning_rate'], history,history_noswap)

wandb.log({"train error plot": wandb.Image(train_plt)})
wandb.log({"test error plot": wandb.Image(test_plt)})
wandb.log({"train error plot(i)": train_plt})
wandb.log({"test error plot(i)": test_plt})
# #
# #
