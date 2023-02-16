import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from keras.models import Sequential


##################################################################################
# Initialization
##################################################################################

# Xavier : tf.initializers.GlorotUniform() or tf.initializers.GlorotNormal()
# He : tf.initializers.he_normal() or tf.initializers.he_uniform()
# Normal : tf.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Truncated_normal : tf.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
# Orthogonal : tf.initializers.Orthogonal0.02)

##################################################################################
# Regularization
##################################################################################

# l2_decay : tf.keras.regularizers.l2(0.0001)
# orthogonal_regularizer : orthogonal_regularizer(0.0001) # orthogonal_regularizer_fully(0.0001)

# factor, mode = pytorch_xavier_weight_factor(gain=0.02, uniform=False)
# distribution = "untruncated_normal"
# distribution in {"uniform", "truncated_normal", "untruncated_normal"}
# weight_initializer = tf.initializers.VarianceScaling(scale=factor, mode=mode, distribution=distribution)

weight_initializer = tf.initializers.RandomNormal(mean=0.0, stddev=0.02)
weight_regularizer = tf.keras.regularizers.l2(0.0001)
weight_regularizer_fully = tf.keras.regularizers.l2(0.0001)

##################################################################################
# Layers
##################################################################################

class Conv(tf.keras.layers.Layer):
    def __init__(self, channels, kernel=3, stride=1, pad=0, pad_type='zero', use_bias=True, sn=False, name='Conv'):
        super(Conv, self).__init__(name=name)
        self.channels = channels
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.pad_type = pad_type
        self.use_bias = use_bias
        self.sn = sn

        if self.sn:
            self.conv = SpectralNormalization(tf.keras.layers.Conv2D(filters=self.channels, kernel_size=self.kernel,
                                                                     kernel_initializer=weight_initializer,
                                                                     kernel_regularizer=weight_regularizer,
                                                                     strides=self.stride, use_bias=self.use_bias),
                                              name='sn_' + self.name)
        else:
            self.conv = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=self.kernel,
                                               kernel_initializer=weight_initializer,
                                               kernel_regularizer=weight_regularizer,
                                               strides=self.stride, use_bias=self.use_bias, name=self.name)

    def call(self, x, training=None, mask=None):
        if self.pad > 0:
            h = x.shape[1]
            if h % self.stride == 0:
                pad = self.pad * 2
            else:
                pad = max(self.kernel - (h % self.stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if self.pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')
            else:
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])

        x = self.conv(x)

        return x


class FullyConnected(tf.keras.layers.Layer):
    def __init__(self, units, use_bias=True, sn=False, name='FullyConnected'):
        super(FullyConnected, self).__init__(name=name)
        self.units = units
        self.use_bias = use_bias
        self.sn = sn

        if self.sn:
            self.fc = SpectralNormalization(tf.keras.layers.Dense(self.units,
                                                                  kernel_initializer=weight_initializer,
                                                                  kernel_regularizer=weight_regularizer_fully,
                                                                  use_bias=self.use_bias), name='sn_' + self.name)
        else:
            self.fc = tf.keras.layers.Dense(self.units,
                                            kernel_initializer=weight_initializer,
                                            kernel_regularizer=weight_regularizer_fully,
                                            use_bias=self.use_bias, name=self.name)

    def call(self, x, training=None, mask=None):
        x = Flatten(x)
        x = self.fc(x)

        return 


def get_residual_layer(res_n) :
    x = []

    if res_n == 18 :
        x = [2, 2, 2, 2]

    if res_n == 34 :
        x = [3, 4, 6, 3]

    if res_n == 50 :
        x = [3, 4, 6, 3]

    if res_n == 101 :
        x = [3, 4, 23, 3]

    if res_n == 152 :
        x = [3, 8, 36, 3]

    return x



##################################################################################
# Blocks
##################################################################################


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, channels, use_bias=True, sn=False, name='ResBlock'):
        super(ResBlock, self).__init__(name=name)
        self.channels = channels
        self.use_bias = use_bias
        self.sn = sn

        self.conv_0 = Conv(self.channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=self.use_bias,
                            sn=self.sn, name='conv_0')
        self.batch_norm_0 = BatchNorm(momentum=0.9, epsilon=1e-5, name='batch_norm_0')

        self.conv_1 = Conv(self.channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=self.use_bias,
                           sn=self.sn, name='conv_1')
        self.batch_norm_1 = BatchNorm(momentum=0.9, epsilon=1e-5, name='batch_norm_1')

    def build(self, input_shape):

        self.skip_flag = self.channels != input_shape[-1]
        if self.skip_flag:
            self.skip_conv = Conv(self.channels, kernel=1, stride=1, use_bias=self.use_bias, sn=self.sn,
                                  name='skip_conv')

    def call(self, x_init, training=None, mask=None):
        with tf.name_scope(self.name):
            with tf.name_scope('res1'):
                x = self.conv_0(x_init)
                x = self.batch_norm_0(x, training=training)
                x = Relu(x)

            with tf.name_scope('res2'):
                x = self.conv_1(x)
                x = self.batch_norm_1(x, training=training)

            if self.skip_flag:
                x_init = self.skip_conv(x_init)
                return Relu(x + x_init)

            else:
                return x + x_init



##################################################################################
# Normalization
##################################################################################

def BatchNorm(momentum=0.9, epsilon=1e-5, name='BatchNorm'):
    return tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon,
                                              center=True, scale=True,
                                              name=name)



##################################################################################
# Activation Function
##################################################################################


def Relu(x=None, name='relu'):
    if x is None:
        return tf.keras.layers.Activation(tf.keras.activations.relu, name=name)

    else:
        return tf.keras.layers.Activation(tf.keras.activations.relu, name=name)(x)



##################################################################################
# Pooling & Resize
##################################################################################

def Global_Avg_Pooling(x=None, name='global_avg_pool'):
    if x is None:
        gap = tf.keras.layers.GlobalAveragePooling2D(name=name)
    else:
        gap = tf.keras.layers.GlobalAveragePooling2D(name=name)(x)
    return gap


def avg_pooling(x, pool_size=2, name='avg_pool'):
    x = tf.keras.layers.AvgPool2D(pool_size=pool_size, strides=pool_size, padding='SAME', name=name)(x)
    return x


def Flatten(x=None, name='flatten'):

    if x is None:
        return tf.keras.layers.Flatten(name=name)
    else :
        return tf.keras.layers.Flatten(name=name)(x)


##################################################################################
# Loss Function
##################################################################################

def classification_loss(logit, label):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit))
    prediction = tf.equal(tf.argmax(logit, -1), tf.argmax(label, -1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return loss, accuracy

    
##################################################################################
# Class function
##################################################################################

class get_weight(tf.keras.layers.Layer):
    def __init__(self, w_shape, w_init, w_regular, w_trainable):
        super(get_weight, self).__init__()

        self.w_shape = w_shape
        self.w_init = w_init
        self.w_regular = w_regular
        self.w_trainable = w_trainable
        # self.w_name = w_name

    def call(self, inputs=None, training=None, mask=None):
        return self.add_weight(shape=self.w_shape, dtype=tf.float32,
                               initializer=self.w_init, regularizer=self.w_regular,
                               trainable=self.w_trainable)


class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape=None):
        self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name=self.name + '_u',
                                 dtype=tf.float32, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        super(SpectralNormalization, self).build()

    def call(self, inputs, training=None, mask=None):
        self.update_weights()
        output = self.layer(inputs)
        # self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
        return output

    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = self.u
        v_hat = None

        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + self.eps)

                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)

        self.layer.kernel = self.w / sigma

    def restore_weights(self):

        self.layer.kernel = self.w
