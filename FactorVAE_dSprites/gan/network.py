import tensorflow as tf
from op import linear, batch_norm, deconv2d, conv2d, lrelu
import os


class Network(object):
    def __init__(self, scope_name):
        self.scope_name = scope_name

    def build(self, input):
        raise NotImplementedError

    @property
    def all_vars(self):
        return tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=self.scope_name)

    @property
    def trainable_vars(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope_name)

    def get_variable(self, name):
        for var in self.all_vars:
            if var.name == name:
                return var
        return None

    def print_layers(self):
        print("Layers of {}".format(self.scope_name))
        print(self.all_vars)

    def save(self, sess, folder):
        saver = tf.train.Saver(self.all_vars)
        path = os.path.join(folder, "model.ckpt")
        saver.save(sess, path)

    def load(self, sess, folder):
        saver = tf.train.Saver(self.all_vars)
        path = os.path.join(folder, "model.ckpt")
        saver.restore(sess, path)


class VAEDecoder(Network):
    def __init__(self, output_width, output_height, output_depth,
                 stride=2, kernel=4,
                 scope_name="VAEDecoder", *args, **kwargs):
        super(VAEDecoder, self).__init__(
            scope_name=scope_name, *args, **kwargs)
        self.output_width = output_width
        self.output_height = output_height
        self.output_depth = output_depth
        self.stride = stride
        self.kernel = kernel

    def build(self, z, train):
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            batch_size = tf.shape(z)[0]

            layers = [z]

            with tf.variable_scope("layer0"):
                layers.append(linear(layers[-1], 128))
                layers.append(tf.nn.relu(layers[-1]))

            with tf.variable_scope("layer1"):
                layers.append(linear(layers[-1], 4 * 4 * 64))
                layers.append(tf.nn.relu(layers[-1]))
                layers.append(tf.reshape(layers[-1], [-1, 4, 4, 64]))

            with tf.variable_scope("layer2"):
                layers.append(deconv2d(
                    layers[-1],
                    [batch_size, 8, 8, 64],
                    d_h=self.stride,
                    d_w=self.stride,
                    k_h=self.kernel,
                    k_w=self.kernel))
                layers.append(tf.nn.relu(layers[-1]))

            with tf.variable_scope("layer3"):
                layers.append(deconv2d(
                    layers[-1],
                    [batch_size, 16, 16, 32],
                    d_h=self.stride,
                    d_w=self.stride,
                    k_h=self.kernel,
                    k_w=self.kernel))
                layers.append(tf.nn.relu(layers[-1]))

            with tf.variable_scope("layer4"):
                layers.append(deconv2d(
                    layers[-1],
                    [batch_size, 32, 32, 32],
                    d_h=self.stride,
                    d_w=self.stride,
                    k_h=self.kernel,
                    k_w=self.kernel))
                layers.append(tf.nn.relu(layers[-1]))

            with tf.variable_scope("layer5"):
                layers.append(deconv2d(
                    layers[-1],
                    [batch_size, self.output_height,
                     self.output_width, self.output_depth],
                    d_h=self.stride,
                    d_w=self.stride,
                    k_h=self.kernel,
                    k_w=self.kernel))

            return layers[-1], layers


class VAEEncoder(Network):
    def __init__(self, output_length,
                 stride=2, kernel=4,
                 scope_name="VAEEncoder", *args, **kwargs):
        super(VAEEncoder, self).__init__(
            scope_name=scope_name, *args, **kwargs)
        self.output_length = output_length
        self.stride = stride
        self.kernel = kernel

    def build(self, image, train):
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            layers = [image]
            with tf.variable_scope("layer0"):
                layers.append(conv2d(
                    layers[-1],
                    32,
                    d_h=self.stride,
                    d_w=self.stride,
                    k_h=self.kernel,
                    k_w=self.kernel))
                layers.append(tf.nn.relu(layers[-1]))
            with tf.variable_scope("layer1"):
                layers.append(conv2d(
                    layers[-1],
                    32,
                    d_h=self.stride,
                    d_w=self.stride,
                    k_h=self.kernel,
                    k_w=self.kernel))
                layers.append(tf.nn.relu(layers[-1]))
            with tf.variable_scope("layer2"):
                layers.append(conv2d(
                    layers[-1],
                    64,
                    d_h=self.stride,
                    d_w=self.stride,
                    k_h=self.kernel,
                    k_w=self.kernel))
                layers.append(tf.nn.relu(layers[-1]))
            with tf.variable_scope("layer3"):
                layers.append(conv2d(
                    layers[-1],
                    64,
                    d_h=self.stride,
                    d_w=self.stride,
                    k_h=self.kernel,
                    k_w=self.kernel))
                layers.append(tf.nn.relu(layers[-1]))
            with tf.variable_scope("layer4"):
                layers.append(linear(layers[-1], 128))
            with tf.variable_scope("layer5-mean"):
                mean = linear(layers[-1], self.output_length)
            with tf.variable_scope("layer5-logvar"):
                logvar = linear(layers[-1], self.output_length)
            layers.append(mean)
            layers.append(logvar)

            return mean, logvar, layers


class MetricRegresser(Network):
    def __init__(self, output_length,
                 stride=2, kernel=4,
                 scope_name="metricRegresser", *args, **kwargs):
        super(MetricRegresser, self).__init__(
            scope_name=scope_name, *args, **kwargs)
        self.output_length = output_length
        self.stride = stride
        self.kernel = kernel

    def build(self, images, train):
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            layers = [images]
            with tf.variable_scope("layer0"):
                layers.append(conv2d(
                    layers[-1],
                    32,
                    d_h=self.stride,
                    d_w=self.stride,
                    k_h=self.kernel,
                    k_w=self.kernel))
                layers.append(lrelu(layers[-1]))
                layers.append(batch_norm()(layers[-1], train=train))
            with tf.variable_scope("layer1"):
                layers.append(conv2d(
                    layers[-1],
                    32,
                    d_h=self.stride,
                    d_w=self.stride,
                    k_h=self.kernel,
                    k_w=self.kernel))
                layers.append(lrelu(layers[-1]))
                layers.append(batch_norm()(layers[-1], train=train))
            with tf.variable_scope("layer2"):
                layers.append(conv2d(
                    layers[-1],
                    64,
                    d_h=self.stride,
                    d_w=self.stride,
                    k_h=self.kernel,
                    k_w=self.kernel))
                layers.append(lrelu(layers[-1]))
                layers.append(batch_norm()(layers[-1], train=train))
            with tf.variable_scope("layer3"):
                layers.append(conv2d(
                    layers[-1],
                    64,
                    d_h=self.stride,
                    d_w=self.stride,
                    k_h=self.kernel,
                    k_w=self.kernel))
                layers.append(lrelu(layers[-1]))
                layers.append(batch_norm()(layers[-1], train=train))
            with tf.variable_scope("layer4"):
                layers.append(linear(layers[-1], 128))
                layers.append(lrelu(layers[-1]))
                layers.append(batch_norm()(layers[-1], train=train))
            with tf.variable_scope("layer5"):
                layers.append(linear(layers[-1], self.output_length))

            return layers[-1], layers


class TCDiscriminator(Network):
    def __init__(self, l_dim=1000, num_layers=6,
                 scope_name="tcDiscriminator", *args, **kwargs):
        super(TCDiscriminator, self).__init__(
            scope_name=scope_name, *args, **kwargs)
        self.l_dim = l_dim
        self.num_layers = num_layers

    def build(self, z, train):
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            layers = [z]

            for i in range(self.num_layers - 1):
                with tf.variable_scope("layer{}".format(i)):
                    layers.append(linear(layers[-1], self.l_dim))
                    layers.append(lrelu(layers[-1]))
            with tf.variable_scope("layer{}".format(self.num_layers - 1)):
                layers.append(linear(layers[-1], 2))
                logit = layers[-1]
                layers.append(tf.nn.softmax(layers[-1]))
                prob = layers[-1]

            return logit, prob, layers
