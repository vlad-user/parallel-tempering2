import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import deep_tempering as dt
from keras.datasets import mnist
from keras.utils import np_utils


def model_builder(hp):
    inputs = tf.keras.layers.Input((2,))
    res = tf.keras.layers.Dense(2, activation=tf.nn.relu)(inputs)
    dropout_rate = hp.get_hparam('dropout_rate', default_value=0.0)
    res = tf.keras.layers.Dropout(dropout_rate)(res)
    res = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(res)
    model = tf.keras.models.Model(inputs, res)

    return model


def mnist_model_builder(hp):
    inputs = tf.keras.layers.Input((28,28,1))
    res = tf.keras.layers.Flatten(input_shape=(28, 28, 1))(inputs)
    res = tf.keras.layers.Dense(128, activation=tf.nn.relu)(res)
    dropout_rate = hp.get_hparam('dropout_rate', default_value=0.0)
    res = tf.keras.layers.Dropout(dropout_rate)(res)
    res = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(res)
    model = tf.keras.models.Model(inputs, res)

    return model


if __name__ == '__main__':
    # (ds_train, ds_test), ds_info = tfds.load(
    #     'mnist',
    #     split=['train', 'test'],
    #     shuffle_files=True,
    #     as_supervised=True,
    #     with_info=True,
    # )
    #
    #
    # def normalize_img(image, label):
    #     return tf.cast(image, tf.float32) / 255., label
    #
    #
    # ds_train = ds_train.map(
    #     normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # ds_train = ds_train.cache()
    # ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    # ds_train = ds_train.batch(128)
    # ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    #
    # ds_test = ds_test.map(
    #     normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # ds_test = ds_test.batch(128)
    # ds_test = ds_test.cache()
    # ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    x = np.random.normal(0, 1, (10, 2))
    y = np.random.randint(0, 2, (10,))

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)

    X_train = X_train / 255
    X_test = X_test / 255
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]



    n_replicas = 6
    model = dt.EnsembleModel(mnist_model_builder)
    hp = {
        'learning_rate': np.linspace(0.01, 0.001, n_replicas),
        'dropout_rate': np.linspace(0, 0.5, n_replicas)
    }

    model.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  n_replicas=n_replicas)


    history = model.fit(X_train,
                        y_train,
                        validation_data=(X_test, y_test),
                        hyper_params=hp,
                        batch_size=2,
                        epochs=10,
                        swap_step=4,
                        burn_in=15)

    # access the optimal (not compiled) keras' model instance
    optimal_model = model.optimal_model()

    # inference only on the trained optimal model
    predicted = optimal_model.predict(x)
