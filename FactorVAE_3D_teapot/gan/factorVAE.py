import tensorflow as tf
import numpy as np
from tqdm import tqdm
import datetime
import imageio
import os
import math
import csv
import copy


class FactorVAE(object):
    # Reference:
    # https://github.com/paruby/FactorVAE/blob/master/factor_vae.py
    def __init__(self, sess, checkpoint_dir, sample_dir, time_path,
                 epoch, batch_size, data,
                 vis_freq, vis_num_sample, vis_num_rep,
                 latent,
                 decoder, encoder, tcDiscriminator,
                 tc_coe,
                 metric_callbacks, metric_freq, metric_path,
                 output_reverse,
                 en_de_lr=0.0001, en_de_beta1=0.9, en_de_beta2=0.999,
                 tcd_lr=0.0001, tcd_beta1=0.5, tcd_beta2=0.9):
        self.sess = sess
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.time_path = time_path
        self.epoch = epoch
        self.batch_size = batch_size
        self.data = data
        self.vis_freq = vis_freq
        self.vis_num_sample = vis_num_sample
        self.vis_num_rep = vis_num_rep
        self.latent = latent
        self.decoder = decoder
        self.encoder = encoder
        self.tcDiscriminator = tcDiscriminator
        self.tc_coe = tc_coe
        self.metric_callbacks = metric_callbacks
        self.metric_freq = metric_freq
        self.metric_path = metric_path
        self.output_reverse = output_reverse
        self.en_de_lr = en_de_lr
        self.en_de_beta1 = en_de_beta1
        self.en_de_beta2 = en_de_beta2
        self.tcd_lr = tcd_lr
        self.tcd_beta1 = tcd_beta1
        self.tcd_beta2 = tcd_beta2

        self.num_images = data.shape[0]
        self.image_dims = list(data.shape[1:])

        self.EPS = 1e-8

        for metric_callback in self.metric_callbacks:
            metric_callback.set_model(self)

        self.vis_latents = self.latent.uniformly_sample(
            vis_num_sample, vis_num_rep)
        self.vis_random_latents = \
            self.latent.sample(vis_num_sample * vis_num_rep)

        if self.latent.in_dim == 0:
            raise NotImplementedError

    def build(self):
        self.build_connection()
        self.build_loss()
        self.build_summary()
        self.build_metric()
        self.saver = tf.train.Saver()

    def build_metric(self):
        for metric_callback in self.metric_callbacks:
            metric_callback.build()

    def build_connection(self):
        self.real_image_pl = tf.placeholder(
            tf.float32, [None] + self.image_dims, name="real_image")

        self.en_mean_train_tf, self.en_logvar_train_tf, _ = \
            self.encoder.build(self.real_image_pl, train=True)
        self.en_mean_test_tf, self.en_logvar_test_tf, _ = \
            self.encoder.build(self.real_image_pl, train=False)

        self.de_input_noise = tf.random_normal(
            shape=tf.shape(self.en_mean_train_tf))
        self.de_input_train_tf = (self.en_mean_train_tf +
                                  ((tf.exp(self.en_logvar_train_tf / 2) *
                                    self.de_input_noise)))
        self.de_input_test_tf = (self.en_mean_test_tf +
                                 ((tf.exp(self.en_logvar_test_tf / 2) *
                                   self.de_input_noise)))
        self.de_train_tf, _ = self.decoder.build(
            self.de_input_train_tf, train=True)

        self.de_shuffled_input_train_tf = self.latent.shuffled_reg_out_tf_var(
            self.de_input_train_tf)

        self.tcd_logit_real_train_tf, self.tcd_prob_real_train_tf, _ = \
            self.tcDiscriminator.build(
                self.de_input_train_tf, train=True)
        self.tcd_logit_fake_train_tf, self.tcd_prob_fake_train_tf, _ = \
            self.tcDiscriminator.build(
                self.de_shuffled_input_train_tf, train=True)

        self.z_pl = tf.placeholder(
            tf.float32, [None, self.latent.in_dim], name="z")
        self.de_test_tf, _ = self.decoder.build(self.z_pl, train=False)
        self.de_test_tf = tf.nn.sigmoid(self.de_test_tf)
        if self.output_reverse:
            self.de_test_tf = 1.0 - self.de_test_tf

        self.decoder.print_layers()
        self.encoder.print_layers()
        self.tcDiscriminator.print_layers()

    def build_loss(self):
        self.real_image_flat_tf = tf.reshape(
            self.real_image_pl, [-1, np.prod(self.image_dims)])
        self.de_train_flat_tf = tf.reshape(
            self.de_train_tf, [-1, np.prod(self.image_dims)])
        self.en_de_loss_recon = tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.de_train_flat_tf,
                    labels=self.real_image_flat_tf),
                axis=1))
        self.en_de_loss_kl = tf.reduce_mean(
            tf.reduce_sum(
                0.5 * (-1.0 -
                       self.en_logvar_train_tf +
                       self.en_mean_train_tf**2 +
                       tf.exp(self.en_logvar_train_tf)),
                axis=1))
        self.en_de_loss_tc = tf.reduce_mean(
            (self.tcd_logit_real_train_tf[:, 0] -
             self.tcd_logit_real_train_tf[:, 1]),
            axis=0)
        self.en_de_loss = (self.en_de_loss_recon +
                           self.en_de_loss_kl +
                           self.tc_coe * self.en_de_loss_tc)

        self.tcd_loss_real = -0.5 * tf.reduce_mean(tf.log(
            self.tcd_prob_real_train_tf[:, 0] + self.EPS))
        self.tcd_loss_fake = -0.5 * tf.reduce_mean(tf.log(
            self.tcd_prob_fake_train_tf[:, 1] + self.EPS))
        self.tcd_loss = self.tcd_loss_real + self.tcd_loss_fake

        self.en_de_op = \
            tf.train.AdamOptimizer(self.en_de_lr, beta1=self.en_de_beta1,
                                   beta2=self.en_de_beta2)\
            .minimize(
                self.en_de_loss,
                var_list=(self.encoder.trainable_vars +
                          self.decoder.trainable_vars))
        self.tcd_op = \
            tf.train.AdamOptimizer(self.tcd_lr, beta1=self.tcd_beta1,
                                   beta2=self.tcd_beta2)\
            .minimize(
                self.tcd_loss,
                var_list=self.tcDiscriminator.trainable_vars)

    def build_summary(self):
        self.en_de_summary = []
        self.en_de_summary.append(tf.summary.scalar(
            "loss/en_de/recon", self.en_de_loss_recon))
        self.en_de_summary.append(tf.summary.scalar(
            "loss/en_de/kl", self.en_de_loss_kl))
        self.en_de_summary.append(tf.summary.scalar(
            "loss/en_de/tc", self.en_de_loss_tc))
        self.en_de_summary.append(tf.summary.scalar(
            "loss/en_de", self.en_de_loss))
        self.en_de_summary = tf.summary.merge(self.en_de_summary)

        self.tcd_summary = []
        self.tcd_summary.append(tf.summary.scalar(
            "loss/tcd/real", self.tcd_loss_real))
        self.tcd_summary.append(tf.summary.scalar(
            "loss/tcd/fake", self.tcd_loss_fake))
        self.tcd_summary.append(tf.summary.scalar(
            "tcd/real", tf.reduce_mean(self.tcd_prob_real_train_tf[:, 0])))
        self.tcd_summary.append(tf.summary.scalar(
            "tcd/fake", tf.reduce_mean(self.tcd_prob_fake_train_tf[:, 0])))
        self.tcd_summary.append(tf.summary.scalar(
            "loss/tcd", self.tcd_loss))
        self.tcd_summary = tf.summary.merge(self.tcd_summary)

    def save(self, global_id, saver=None, checkpoint_dir=None):
        if saver is None:
            saver = self.saver
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir
        saver.save(
            self.sess,
            os.path.join(checkpoint_dir, "model"),
            global_step=global_id)

    def load(self, checkpoint_dir=None):
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def _image_list_to_grid(self, image_list, num_row, num_col):
        assert num_row * num_col == image_list.shape[0]

        height, width, depth = self.image_dims
        image = np.zeros((num_row * height,
                          num_col * width,
                          depth))
        s_id = 0
        for row in range(num_row):
            for col in range(num_col):
                image[row * height: (row + 1) * height,
                      col * width: (col + 1) * width, :] = image_list[s_id]
                s_id += 1

        v_min = image.min() - self.EPS
        v_max = image.max() + self.EPS
        image = (image - v_min) / (v_max - v_min) * 255.0
        image = image.astype(np.uint8)

        print(v_min, v_max)

        return image

    def sample_from(self, z):
        # print(z.shape)
        samples = []
        for i in range(int(math.ceil(float(z.shape[0]) / self.batch_size))):
            sub_samples = self.sess.run(
                self.de_test_tf,
                feed_dict={self.z_pl: z[i * self.batch_size:
                                        (i + 1) * self.batch_size]})
            samples.append(sub_samples)
        return np.vstack(samples)

    def inference_from(self, img):
        latents = []
        for i in range(int(math.ceil(float(img.shape[0]) / self.batch_size))):
            sub_latents = self.sess.run(
                self.de_input_test_tf,
                feed_dict={self.real_image_pl: img[i * self.batch_size:
                                                   (i + 1) * self.batch_size]})
            latents.append(sub_latents)
        return np.vstack(latents)

    def visualize(self, epoch_id, batch_id, global_id):
        for i, latent in enumerate(self.vis_latents):
            samples = self.sample_from(latent)
            # print(samples.shape)
            image = self._image_list_to_grid(
                samples, self.vis_num_sample, self.vis_num_rep)
            file_path = os.path.join(
                self.sample_dir,
                "epoch_id-{},batch_id-{},global_id-{},latent-{}.png".format(
                    epoch_id, batch_id, global_id, i))
            imageio.imwrite(file_path, image)

        samples = self.sample_from(self.vis_random_latents)
        image = self._image_list_to_grid(
            samples, self.vis_num_sample, self.vis_num_rep)
        file_path = os.path.join(
            self.sample_dir,
            "epoch_id-{},batch_id-{},global_id-{},latent-all.png".format(
                epoch_id, batch_id, global_id))
        imageio.imwrite(file_path, image)

        recovered_vis_random_latents = self.inference_from(samples)
        file_path = os.path.join(
            self.sample_dir,
            "epoch_id-{},batch_id-{},global_id-{},latent-recovered.npz".format(
                epoch_id, batch_id, global_id))
        np.savez(
            file_path,
            vis_random_latents=self.vis_random_latents,
            recovered_vis_random_latents=recovered_vis_random_latents,
            samples=samples)

    def log_metric(self, epoch_id, batch_id, global_id):
        if self.metric_callbacks is not None:
            metric = {}
            for metric_callback in self.metric_callbacks:
                metric.update(metric_callback.evaluate(
                    epoch_id, batch_id, global_id))
            if not os.path.isfile(self.metric_path):
                self.METRIC_FIELD_NAMES = ["epoch_id", "batch_id", "global_id"]
                for k in metric:
                    self.METRIC_FIELD_NAMES.append(k)
                with open(self.metric_path, "wb") as csv_file:
                    writer = csv.DictWriter(
                        csv_file, fieldnames=self.METRIC_FIELD_NAMES)
                    writer.writeheader()
            with open(self.metric_path, "ab") as csv_file:
                writer = csv.DictWriter(
                    csv_file, fieldnames=self.METRIC_FIELD_NAMES)
                data = {
                    "epoch_id": epoch_id,
                    "batch_id": batch_id,
                    "global_id": global_id}
                metric_string = copy.deepcopy(metric)
                for k in metric_string:
                    if isinstance(metric[k], (float, np.float32, np.float64)):
                        metric_string[k] = "{0:.12f}".format(metric_string[k])
                data.update(metric_string)
                writer.writerow(data)
            for k in metric:
                if isinstance(metric[k], (int, long, float, complex,
                                          np.float32, np.float64)):
                    summary = tf.Summary(
                        value=[tf.Summary.Value(
                            tag="metric/" + k, simple_value=metric[k])])
                    self.summary_writer.add_summary(summary, global_id)

    def train(self):
        tf.global_variables_initializer().run()
        self.summary_writer = tf.summary.FileWriter(
            self.checkpoint_dir, self.sess.graph)

        for metric_callback in self.metric_callbacks:
            metric_callback.load()

        batch_num = len(self.data) // self.batch_size

        global_id = 0

        data_id = np.arange(self.data.shape[0])

        for epoch_id in tqdm(range(self.epoch)):
            np.random.shuffle(data_id)

            with open(self.time_path, "a") as f:
                time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                f.write("epoch {} starts: {}\n".format(epoch_id, time))

            for batch_id in range(batch_num):
                batch_data_id = \
                    data_id[batch_id * self.batch_size:
                            (batch_id + 1) * self.batch_size]
                batch_image = self.data[batch_data_id]

                feed_dict = {self.real_image_pl: batch_image}

                summary_result, _ = self.sess.run(
                    [self.en_de_summary, self.en_de_op],
                    feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_result, global_id)

                summary_result, _ = self.sess.run(
                    [self.tcd_summary, self.tcd_op],
                    feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_result, global_id)

                if (batch_id + 1) % self.vis_freq == 0:
                    self.visualize(epoch_id, batch_id, global_id)

                if (batch_id + 1) % self.metric_freq == 0:
                    self.log_metric(epoch_id, batch_id, global_id)

                global_id += 1

            self.visualize(epoch_id, -1, global_id - 1)
            self.log_metric(epoch_id, -1, global_id - 1)

            with open(self.time_path, "a") as f:
                time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                f.write("epoch {} ends: {}\n".format(epoch_id, time))

            self.save(global_id - 1)


if __name__ == "__main__":
    from latent import GaussianLatent, JointLatent
    from network import VAEDecoder, VAEEncoder, TCDiscriminator, \
        MetricRegresser
    from load_data import load_3Dpots
    from metric import FactorVAEMetric

    data, metric_data, latent_values = \
        load_3Dpots("../data/3Dpots")
    _, height, width, depth = data.shape
    print(data.min(), data.max())
    
    latent_list = []
    for i in range(10):
        latent_list.append(
            GaussianLatent(
                in_dim=1, out_dim=1, apply_reg=True))
    latent = JointLatent(latent_list=latent_list)

    decoder = VAEDecoder(
        output_width=width, output_height=height, output_depth=depth)
    encoder = VAEEncoder(output_length=latent.reg_in_dim)
    tcDiscriminator = TCDiscriminator()

    checkpoint_dir = "./test/checkpoint"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    sample_dir = "./test/sample"
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    time_path = "./test/time.txt"
    metric_path = "./test/metric.csv"
    epoch = 35
    batch_size = 64
    vis_freq = 200
    vis_num_sample = 10
    vis_num_rep = 10
    metric_freq = 400
    tc_coe = 40.0

    run_config = tf.ConfigProto()
    with tf.Session(config=run_config) as sess:
        factorVAEMetric = FactorVAEMetric(metric_data, sess=sess)
        vae = FactorVAE(
            sess=sess,
            checkpoint_dir=checkpoint_dir,
            sample_dir=sample_dir,
            time_path=time_path,
            epoch=epoch,
            batch_size=batch_size,
            data=data,
            vis_freq=vis_freq,
            vis_num_sample=vis_num_sample,
            vis_num_rep=vis_num_rep,
            latent=latent,
            decoder=decoder,
            encoder=encoder,
            tcDiscriminator=tcDiscriminator,
            tc_coe=tc_coe,
            metric_callbacks=[factorVAEMetric],
            metric_freq=metric_freq,
            metric_path=metric_path,
            output_reverse=False)
        vae.build()
        vae.train()
