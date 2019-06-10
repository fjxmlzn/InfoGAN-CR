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


class Decoder(Network):
    def __init__(self, output_width, output_height, output_depth,
                 stride=2, kernel=4,
                 scope_name="decoder", *args, **kwargs):
        super(Decoder, self).__init__(scope_name=scope_name, *args, **kwargs)
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
                layers.append(batch_norm()(layers[-1], train=train))

            with tf.variable_scope("layer1"):
                layers.append(linear(layers[-1], 4 * 4 * 64))
                layers.append(tf.nn.relu(layers[-1]))
                layers.append(batch_norm()(layers[-1], train=train))
                layers.append(tf.reshape(layers[-1], [-1, 4, 4, 64]))

            with tf.variable_scope("layer2"):
                layers.append(deconv2d(
                    layers[-1],
                    [batch_size, 8, 8, 64],
                    d_h=self.stride,
                    d_w=self.stride,
                    k_h=self.kernel,
                    k_w=self.kernel))
                layers.append(lrelu(layers[-1]))
                layers.append(batch_norm()(layers[-1], train=train))

            with tf.variable_scope("layer3"):
                layers.append(deconv2d(
                    layers[-1],
                    [batch_size, 16, 16, 32],
                    d_h=self.stride,
                    d_w=self.stride,
                    k_h=self.kernel,
                    k_w=self.kernel))
                layers.append(lrelu(layers[-1]))
                layers.append(batch_norm()(layers[-1], train=train))

            with tf.variable_scope("layer4"):
                layers.append(deconv2d(
                    layers[-1],
                    [batch_size, 32, 32, 32],
                    d_h=self.stride,
                    d_w=self.stride,
                    k_h=self.kernel,
                    k_w=self.kernel))
                layers.append(lrelu(layers[-1]))
                layers.append(batch_norm()(layers[-1], train=train))

            with tf.variable_scope("layer5"):
                layers.append(deconv2d(
                    layers[-1],
                    [batch_size, self.output_height,
                     self.output_width, self.output_depth],
                    d_h=self.stride,
                    d_w=self.stride,
                    k_h=self.kernel,
                    k_w=self.kernel))
                layers.append(tf.nn.sigmoid(layers[-1]))

            return layers[-1], layers


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


class CrDiscriminator(Network):
    def __init__(self, output_length,
                 stride=2, kernel=4,
                 scope_name="crDiscriminator", *args, **kwargs):
        super(CrDiscriminator, self).__init__(
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
                    k_w=self.kernel,
                    sn_op=None))
                layers.append(lrelu(layers[-1]))
            with tf.variable_scope("layer1"):
                layers.append(conv2d(
                    layers[-1],
                    32,
                    d_h=self.stride,
                    d_w=self.stride,
                    k_h=self.kernel,
                    k_w=self.kernel,
                    sn_op=None))
                layers.append(lrelu(layers[-1]))
                layers.append(batch_norm()(layers[-1], train=train))
            with tf.variable_scope("layer2"):
                layers.append(conv2d(
                    layers[-1],
                    64,
                    d_h=self.stride,
                    d_w=self.stride,
                    k_h=self.kernel,
                    k_w=self.kernel,
                    sn_op=None))
                layers.append(lrelu(layers[-1]))
                layers.append(batch_norm()(layers[-1], train=train))
            with tf.variable_scope("layer3"):
                layers.append(conv2d(
                    layers[-1],
                    64,
                    d_h=self.stride,
                    d_w=self.stride,
                    k_h=self.kernel,
                    k_w=self.kernel,
                    sn_op=None))
                layers.append(lrelu(layers[-1]))
                layers.append(batch_norm()(layers[-1], train=train))
            with tf.variable_scope("layer4"):
                layers.append(linear(layers[-1], 128, sn_op=None))
                layers.append(lrelu(layers[-1]))
                layers.append(batch_norm()(layers[-1], train=train))
            with tf.variable_scope("layer5"):
                layers.append(linear(layers[-1], self.output_length))

            return layers[-1], layers


class InfoGANDiscriminator(Network):
    def __init__(self, output_length,
                 stride=2, kernel=4, q_l_dim=128,
                 scope_name="infoGANDiscriminator", *args, **kwargs):
        super(InfoGANDiscriminator, self).__init__(
            scope_name=scope_name, *args, **kwargs)
        self.output_length = output_length
        self.stride = stride
        self.kernel = kernel
        self.q_l_dim = q_l_dim

    def build(self, image, train, sn_op):
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("shared"):
                layers = [image]
                with tf.variable_scope("layer0"):
                    layers.append(conv2d(
                        layers[-1],
                        32,
                        d_h=self.stride,
                        d_w=self.stride,
                        k_h=self.kernel,
                        k_w=self.kernel,
                        sn_op=sn_op))
                    layers.append(lrelu(layers[-1]))
                with tf.variable_scope("layer1"):
                    layers.append(conv2d(
                        layers[-1],
                        32,
                        d_h=self.stride,
                        d_w=self.stride,
                        k_h=self.kernel,
                        k_w=self.kernel,
                        sn_op=sn_op))
                    layers.append(lrelu(layers[-1]))
                with tf.variable_scope("layer2"):
                    layers.append(conv2d(
                        layers[-1],
                        64,
                        d_h=self.stride,
                        d_w=self.stride,
                        k_h=self.kernel,
                        k_w=self.kernel,
                        sn_op=sn_op))
                    layers.append(lrelu(layers[-1]))
                with tf.variable_scope("layer3"):
                    layers.append(conv2d(
                        layers[-1],
                        64,
                        d_h=self.stride,
                        d_w=self.stride,
                        k_h=self.kernel,
                        k_w=self.kernel,
                        sn_op=sn_op))
                    layers.append(lrelu(layers[-1]))
                with tf.variable_scope("layer4"):
                    layers.append(linear(layers[-1], 128, sn_op=sn_op))
                    layers.append(lrelu(layers[-1]))

            with tf.variable_scope("x"):
                image_layers = [layers[-1]]
                with tf.variable_scope("layer0"):
                    image_layers.append(
                        linear(image_layers[-1], 1, sn_op=sn_op))
                    image_layers.append(tf.nn.sigmoid(image_layers[-1]))

            with tf.variable_scope("q"):
                q_layers = [layers[-1]]
                with tf.variable_scope("layer0"):
                    q_layers.append(linear(
                        q_layers[-1], self.q_l_dim, sn_op=sn_op))
                    q_layers.append(lrelu(q_layers[-1]))
                with tf.variable_scope("layer1"):
                    q_layers.append(linear(
                        q_layers[-1], self.output_length, sn_op=sn_op))

            layers.extend(image_layers)
            layers.extend(q_layers)

            return image_layers[-1], q_layers[-1], layers
