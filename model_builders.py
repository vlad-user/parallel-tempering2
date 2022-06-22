import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.applications import ResNet50, DenseNet121
from tensorflow.python.keras.backend import int_shape
from tensorflow.keras.models import Model

from deep_tempering.training_utils import get_training_phase_placeholder


class AvgPoolWithWeights(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding="valid", data_format=None, activation=None, **kwargs):
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format

        self.activation = activation
        super(AvgPoolWithWeights, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=self.pool_size, strides=self.strides,
                                                         padding=self.padding, data_format=self.data_format)
        w_init = tf.contrib.layers.xavier_initializer()

        self.w = self.add_weight(name='w',
                                      shape=(input_shape.as_list()[-1],),
                                      initializer=w_init,
                                      trainable=True, dtype=tf.float32)

        self.b = self.add_weight(name='b',
                                 shape=(input_shape.as_list()[-1],),
                                 initializer='zeros',
                                 trainable=True, dtype=tf.float32)

        super(AvgPoolWithWeights, self).build(input_shape)

    def call(self, x, **kwargs):
        x = self.avg_pool(x)
        x = x * self.w + self.b
        if self.activation:
            return self.activation(x)
        return x


    # def compute_output_shape(self, input_shape):
    #     return input_shape


class RBFEuclidean(tf.keras.layers.Layer):
    def __init__(self, units=10, activation=None, **kwargs):
        self.units = units
        self.activation = activation
        super(RBFEuclidean, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        w_init = tf.contrib.layers.xavier_initializer()
        self.w = self.add_weight(name='w',
                                 shape=(input_shape.as_list()[1], self.units),
                                 initializer=w_init,
                                 trainable=True, dtype=tf.float32)
        super(RBFEuclidean, self).build(input_shape)

    def call(self, x, **kwargs):
        x = tf.reduce_sum(tf.square(x[..., None] - self.w), axis=1)
        if self.activation:
            return self.activation(x)
        return x


class AugmentImages(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AugmentImages, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AugmentImages, self).build(input_shape)

    # def augment(self, x):
    #     with tf.device('cpu:0'):
    #         maybe_flipped = tf.image.random_flip_left_right(x)
    #         padded = tf.pad(maybe_flipped, [[0, 0], [4, 4], [4, 4], [0, 0]])
    #         cropped = tf.image.random_crop(padded, size=tf.shape(x))
    #     return cropped

    def call(self, x, training=None, **kwargs):
        shape = tf.shape(x)
        if not training:
            return x
        # if tf.keras.backend.learning_phase():
        #     with tf.device('cpu:0'):
        x = tf.image.random_flip_left_right(x)
        x = tf.pad(x, [[0, 0], [4, 4], [4, 4], [0, 0]])
        x = tf.image.random_crop(x, size=shape)

        return x

class CustomDropout(tf.keras.layers.Layer):
    def __init__(self, drop_prob, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.rate = 1 - drop_prob

    def build(self, input_shape):
        super(CustomDropout, self).build(input_shape)

    def call(self, x, training):
        if training:
            x = tf.nn.dropout(x, self.rate)
        return x


class MyInit(tf.keras.initializers.Initializer):

    def __init__(self, mean, stddev):
      self.mean = mean
      self.stddev = stddev

    def __call__(self, shape, dtype=None):
      return tf.random.normal(shape, mean=self.mean, stddev=self.stddev, dtype=dtype)

    def get_config(self):  # To support serialization
        return {'mean': self.mean, 'stddev': self.stddev}



def lenet5_emnist_builder(hp):

    is_training = get_training_phase_placeholder()
    dropout_rate = hp.get_hparam('dropout_rate', default_value=0.0)


    inputs = tf.keras.layers.Input((32,32,1))
    res = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='tanh',)(inputs)
    res = AvgPoolWithWeights(pool_size=(2, 2,), strides=(2, 2,), activation=tf.keras.activations.tanh)(res)
    res = CustomDropout(dropout_rate)(res, training=is_training)
    res = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='tanh')(res)
    res = AvgPoolWithWeights(pool_size=(2, 2,), strides=(2, 2), activation=tf.keras.activations.tanh)(res)
    res = CustomDropout(dropout_rate)(res, training=is_training)

    res = tf.keras.layers.Conv2D(filters=120, kernel_size=(5, 5), activation='tanh')(res)
    res = CustomDropout(dropout_rate)(res, training=is_training)

    res = tf.keras.layers.Flatten()(res)
    res = CustomDropout(dropout_rate)(res, training=is_training)
    res = tf.keras.layers.Dense(units=84, activation='tanh')(res)
    res = CustomDropout(dropout_rate)(res, training=is_training)

    res = RBFEuclidean(units=26, activation=tf.keras.activations.softmax)(res)
    model = tf.keras.models.Model(inputs, res)
    return model

def lenet5_emnist_builder_2(hp):

    dropout_rate = hp.get_hparam('dropout_rate', default_value=0.)

    inputs = tf.keras.layers.Input((32,32,1))

    res = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1,1), activation='relu',)(inputs)
    res = tf.keras.layers.MaxPooling2D(pool_size=(2, 2,), strides=(2, 2,))(res)
    # res = tf.keras.layers.Lambda(lambda x: tf.nn.dropout(x[0], x[1]))()
    res = CustomDropout(dropout_rate)(res)


    res = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(res)
    res = tf.keras.layers.MaxPooling2D(pool_size=(2, 2,), strides=(2, 2,))(res)

    res = CustomDropout(dropout_rate)(res)

    res = tf.keras.layers.Conv2D(filters=120, kernel_size=(5, 5), activation='relu')(res)
    res = tf.keras.layers.Flatten()(res)

    res = CustomDropout(dropout_rate)(res)
    res = tf.keras.layers.Dense(units=84, activation='relu')(res)

    res = CustomDropout(dropout_rate)(res)
    res = tf.keras.layers.Dense(units=26, activation='softmax')(res) #ToDo: check whther softmax is present in research code or pass params to loss to work with logits
    model = tf.keras.models.Model(inputs, res)
    return model


def lenet5_cifar10_builder(hp):
    is_training = get_training_phase_placeholder()

    dropout_rate = hp.get_hparam('dropout_rate', default_value=0.)

    inputs = tf.keras.layers.Input((32,32,3))

    res = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1,1), activation='relu',)(inputs)
    res = tf.keras.layers.MaxPooling2D(pool_size=(2, 2,), strides=(2, 2,))(res)
    # res = tf.keras.layers.Lambda(lambda x: tf.nn.dropout(x[0], x[1]))()
    res = CustomDropout(dropout_rate)(res, training=is_training)


    res = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(res)
    res = tf.keras.layers.MaxPooling2D(pool_size=(2, 2,), strides=(2, 2,))(res)

    res = CustomDropout(dropout_rate)(res, training=is_training)

    res = tf.keras.layers.Conv2D(filters=120, kernel_size=(5, 5), activation='relu')(res)
    res = tf.keras.layers.Flatten()(res)

    res = CustomDropout(dropout_rate)(res, training=is_training)
    res = tf.keras.layers.Dense(units=84, activation='relu')(res)

    res = CustomDropout(dropout_rate)(res, training=is_training)
    res = tf.keras.layers.Dense(units=10, activation='softmax')(res) #ToDo: check whther softmax is present in research code or pass params to loss to work with logits
    model = tf.keras.models.Model(inputs, res)
    return model

def lenet5_cifar10_same_init_builder(*args):
    s = np.random.get_state()[1][0]
    print(s)
    initializer = tf.keras.initializers.glorot_uniform(s)

    hp = args[0] if len(args) > 0 else None
    if hp:
        dropout_rate = hp.get_hparam('dropout_rate', default_value=0.0)
    else:
        dropout_rate = 0.4

    # dropout_rate = 0.4

    inputs = tf.keras.layers.Input((32,32,3))

    res = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1,1), activation='relu', kernel_initializer=initializer)(inputs)
    res = tf.keras.layers.MaxPooling2D(pool_size=(2, 2,), strides=(2, 2,))(res)
    res = CustomDropout(dropout_rate)(res)


    res = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu', kernel_initializer=initializer)(res)
    res = tf.keras.layers.MaxPooling2D(pool_size=(2, 2,), strides=(2, 2,))(res)

    res = CustomDropout(dropout_rate)(res)

    res = tf.keras.layers.Conv2D(filters=120, kernel_size=(5, 5), activation='relu', kernel_initializer=initializer)(res)
    res = tf.keras.layers.Flatten()(res)

    res = CustomDropout(dropout_rate)(res)
    res = tf.keras.layers.Dense(units=84, activation='relu', kernel_initializer=initializer)(res)

    res = CustomDropout(dropout_rate)(res)
    res = tf.keras.layers.Dense(units=10, activation='softmax', kernel_initializer=initializer)(res)
    model = tf.keras.models.Model(inputs, res)
    return model



def lenet5_cifar10_with_augmentation_builder(hp):

    dropout_rate = hp.get_hparam('dropout_rate', default_value=0.0)

    inputs = tf.keras.layers.Input((32,32,3))
    res = AugmentImages()(inputs)
    res = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1,1), activation='relu',)(res)
    res = tf.keras.layers.MaxPooling2D(pool_size=(2, 2,), strides=(2, 2,))(res)
    res = tf.keras.layers.Dropout(dropout_rate)(res)

    res = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(res)
    res = tf.keras.layers.MaxPooling2D(pool_size=(2, 2,), strides=(2, 2,))(res)
    res = tf.keras.layers.Dropout(dropout_rate)(res)


    res = tf.keras.layers.Conv2D(filters=120, kernel_size=(5, 5), activation='relu')(res)
    res = tf.keras.layers.Flatten()(res)
    res = tf.keras.layers.Dropout(dropout_rate)(res)

    res = tf.keras.layers.Dense(units=84, activation='relu')(res)
    res = tf.keras.layers.Dropout(dropout_rate)(res)

    res = tf.keras.layers.Dense(units=10, activation='softmax')(res) #ToDo: check whther softmax is present in research code or pass params to loss to work with logits
    model = tf.keras.models.Model(inputs, res)
    return model

def resnet50_cifar10(hp):
    dropout_rate = hp.get_hparam('dropout_rate', default_value=0.0)

    base_model = ResNet50(
        weights='imagenet',
        input_shape=(32, 32, 3),
        include_top=False)
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))

    res = base_model(inputs, training=True)
    res = tf.keras.layers.GlobalAveragePooling2D()(res)
    res = tf.keras.layers.Dense(units=10, activation='softmax')(res)
    model = tf.keras.Model(inputs, res)
    return model

def densenet121_cifar10(hp):
    dropout_rate = hp.get_hparam('dropout_rate', default_value=0.0)

    base_model = DenseNet121(
        weights=None,
        input_shape=(32, 32, 3),
        include_top=False)
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))

    res = base_model(inputs, training=True)
    res = tf.keras.layers.GlobalAveragePooling2D()(res)
    res = tf.keras.layers.Dense(units=10, activation='softmax')(res)
    model = tf.keras.Model(inputs, res)
    return model

def dense_block(x, blocks, name):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_relu')(x)
    x = tf.keras.layers.Conv2D(int(int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = tf.keras.layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        Output tensor for the block.
    """
    bn_axis = 3
    x1 = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = tf.keras.layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = tf.keras.layers.Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = tf.keras.layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = tf.keras.layers.Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = tf.keras.layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x



def densenet121_dropout_cifar10_aug(*args):
    # global backend, layers, models, keras_utils
    # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)


    # Determine proper input shape
    hp = args[0] if len(args) > 0 else None
    if hp:
        dropout_rate = hp.get_hparam('dropout_rate', default_value=0.0)
    else:
        dropout_rate = 0.2

    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    x = AugmentImages()(inputs)
    blocks = [6, 12, 24, 16]


    bn_axis = 3

    x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(x)
    x = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = tf.keras.layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = tf.keras.layers.Activation('relu', name='conv1/relu')(x)
    x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = CustomDropout(dropout_rate)(x)

    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = CustomDropout(dropout_rate)(x)

    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = CustomDropout(dropout_rate)(x)

    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')
    x = CustomDropout(dropout_rate)(x)


    x = tf.keras.layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = tf.keras.layers.Activation('relu', name='relu')(x)


    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dense(10, activation='softmax', name='fc1000')(x)


    model = tf.keras.Model(inputs, x)
    #
    alpha = 0.001  # weight decay coefficient

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(tf.keras.regularizers.l2(alpha)(layer.kernel))
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(tf.keras.regularizers.l2(alpha)(layer.bias))

    # BASE_WEIGTHS_PATH = (
    #     'https://github.com/keras-team/keras-applications/'
    #     'releases/download/densenet/')
    # DENSENET121_WEIGHT_PATH = (
    #         BASE_WEIGTHS_PATH +
    #         'densenet121_weights_tf_dim_ordering_tf_kernels.h5')
    # DENSENET121_WEIGHT_PATH_NO_TOP = (
    #         BASE_WEIGTHS_PATH +
    #         'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
    # weights_path = get_file(
    #     'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #     DENSENET121_WEIGHT_PATH_NO_TOP,
    #     cache_subdir='models',
    #     file_hash='30ee3e1110167f948a6b9946edeeb738')
    #
    # model.load_weights(weights_path)


    return model


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 is_training=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization(axis=3, trainable=True)(x, training=is_training, ) # add regularization to batch?
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization(axis=3, trainable=True)(x, training=is_training, )
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v2_20_cifar10(hp):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    depth = 2 * 9 + 2
    input_shape = (32, 32, 3)
    num_classes = 10
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = AugmentImages()(inputs)


    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=x,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tf.keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
    y = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def resnet20_v1_cifar10(*args):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    hp = args[0] if len(args) > 0 else None
    if hp:
        dropout_rate = hp.get_hparam('dropout_rate', default_value=0.0)
        is_training = get_training_phase_placeholder()
        # is_training = True
    else:
        is_training = tf.keras.backend.learning_phase()
        # is_training = False

    n = 3
    depth = n * 6 + 2
    input_shape = (32, 32, 3)
    num_classes = 10
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = AugmentImages()(inputs, training=is_training)
    x = resnet_layer(inputs=x, is_training=is_training)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides,
                             is_training=is_training)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None,
                             is_training=is_training)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 is_training=is_training)
            x = tf.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def resnet56_v1_cifar(*args):
    hp = args[0] if len(args) > 0 else None
    if hp:
        dropout_rate = hp.get_hparam('dropout_rate', default_value=0.0)

    n = 9
    depth = n * 6 + 2
    input_shape = (32, 32, 3)
    num_classes = 10
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = AugmentImages()(inputs)
    x = resnet_layer(inputs=x)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tf.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v1_cifar(*args):
    hp = args[0] if len(args) > 0 else None
    if hp:
        dropout_rate = hp.get_hparam('dropout_rate', default_value=0.0)

    n = 9
    depth = n * 6 + 2
    input_shape = (32, 32, 3)
    num_classes = 10
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = AugmentImages()(inputs)
    x = resnet_layer(inputs=x)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tf.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def resnet121_v1_cifar(*args):
    hp = args[0] if len(args) > 0 else None
    if hp:
        dropout_rate = hp.get_hparam('dropout_rate', default_value=0.0)

    n = 18
    depth = n * 6 + 2
    input_shape = (32, 32, 3)
    num_classes = 10
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = AugmentImages()(inputs)
    x = resnet_layer(inputs=x)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tf.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def xception_cifar10(*args):
    input_shape = (32, 32, 3)
    num_classes = 10

    hp = args[0] if len(args) > 0 else None
    if hp:
        # print(' ##########################################################33')
        # print(hp.get_hparam('learning_rate', default_value=0.00000000))
        # print(' ##########################################################33')

        dropout_rate = hp.get_hparam('dropout_rate', default_value=0.0)
    else:
        dropout_rate = 0

    # data_augmentation = tf.keras.Sequential(
    #     [tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.1),]
    # )

    inputs = tf.keras.Input(shape=input_shape)
    # Image augmentation block
    x = AugmentImages()(inputs)

    # Entry block
    # x =tf.keras.layers.Rescaling(1.0 / 255)(x)
    x =tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x =tf.keras.layers.BatchNormalization()(x)
    x =tf.keras.layers.Activation("relu")(x)

    x =tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x =tf.keras.layers.BatchNormalization()(x)
    x =tf.keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x =tf.keras.layers.Activation("relu")(x)
        x =tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x =tf.keras.layers.BatchNormalization()(x)

        x =tf.keras.layers.Activation("relu")(x)
        x =tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x =tf.keras.layers.BatchNormalization()(x)

        x =tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual =tf.keras.layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x =tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x =tf.keras.layers.SeparableConv2D(1024, 3, padding="same")(x)
    x =tf.keras.layers.BatchNormalization()(x)
    x =tf.keras.layers.Activation("relu")(x)

    x =tf.keras.layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    # x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs =tf.keras.layers.Dense(units, activation=activation)(x)
    return Model(inputs, outputs)

def fc_cifar10(*args):
    input_shape = (32, 32, 3)
    num_classes = 10

    hp = args[0] if len(args) > 0 else None
    if hp:
        # print(' ##########################################################33')
        # print(hp.get_hparam('learning_rate', default_value=0.00000000))
        # print(' ##########################################################33')

        dropout_rate = hp.get_hparam('dropout_rate', default_value=0.0)
    else:
        dropout_rate = 0

    # data_augmentation = tf.keras.Sequential(
    #     [tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.1),]
    # )

    inputs = tf.keras.Input(shape=input_shape)
    # Image augmentation block
    x = AugmentImages()(inputs)
    x = Flatten()(x)
    x =tf.keras.layers.Dense(1024, activation='relu')(x)
    x =tf.keras.layers.Dense(512, activation='relu')(x)
    x =tf.keras.layers.Dense(256, activation='relu')(x)



    outputs =tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)


def simple_cnn_cifar10(*args):
    input_shape = (32, 32, 3)
    num_classes = 10

    hp = args[0] if len(args) > 0 else None
    if hp:
        # print(' ##########################################################33')
        # print(hp.get_hparam('learning_rate', default_value=0.00000000))
        # print(' ##########################################################33')

        dropout_rate = hp.get_hparam('dropout_rate', default_value=0.0)
        is_training = get_training_phase_placeholder()

        # is_training = True
    else:
        dropout_rate = 0
        is_training = True
    inputs = tf.keras.Input(shape=input_shape)
    x = AugmentImages()(inputs, is_training)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)


















