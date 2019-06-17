import tensorflow as tf
import numpy as np
from tqdm import tqdm
import datetime
import imageio
import os
import math
import csv
from sn import NO_OPS
import copy


class INFOGAN_CR(object):
    def __init__(self, sess, checkpoint_dir, sample_dir, time_path,
                 epoch, batch_size, data,
                 vis_freq, vis_num_sample, vis_num_rep,
                 latent,
                 decoder, infoGANDiscriminator, crDiscriminator,
                 gap_start, gap_decrease_times,
                 gap_decrease, gap_decrease_batch,
                 cr_coe_start, cr_coe_increase_times,
                 cr_coe_increase, cr_coe_increase_batch,
                 info_coe_de, info_coe_infod,
                 metric_callbacks, metric_freq, metric_path,
                 output_reverse,
                 summary_freq=1,
                 de_lr=0.001, de_beta1=0.5,
                 infod_lr=0.002, infod_beta1=0.5,
                 crd_lr=0.002, crd_beta1=0.5):
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
        self.infoGANDiscriminator = infoGANDiscriminator
        self.crDiscriminator = crDiscriminator
        self.gap_start = gap_start
        self.gap_decrease_times = gap_decrease_times
        self.gap_decrease = gap_decrease
        self.gap_decrease_batch = gap_decrease_batch
        self.cr_coe_start = cr_coe_start
        self.cr_coe_increase_times = cr_coe_increase_times
        self.cr_coe_increase = cr_coe_increase
        self.cr_coe_increase_batch = cr_coe_increase_batch
        self.info_coe_de = info_coe_de
        self.info_coe_infod = info_coe_infod
        self.metric_callbacks = metric_callbacks
        self.metric_freq = metric_freq
        self.metric_path = metric_path
        self.output_reverse = output_reverse
        self.de_lr = de_lr
        self.de_beta1 = de_beta1
        self.infod_lr = infod_lr
        self.infod_beta1 = infod_beta1
        self.crd_lr = crd_lr
        self.crd_beta1 = crd_beta1

        self.num_images = data.shape[0]
        self.image_dims = list(data.shape[1:])

        self.summary_freq = summary_freq

        self.EPS = 1e-8
        self.SN_OP = "spectral_norm_update_ops"

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
        self.z_pl = tf.placeholder(
            tf.float32, [None, self.latent.in_dim], name="z")
        self.z_cr_pl_l = [
            tf.placeholder(
                tf.float32, [None, self.latent.in_dim], name="z_cr1"),
            tf.placeholder(
                tf.float32, [None, self.latent.in_dim], name="z_cr2")]
        self.real_image_pl = tf.placeholder(
            tf.float32, [None] + self.image_dims, name="real_image")
        self.train_with_crd_pl = tf.placeholder(
            tf.bool, (), name="train_with_crd")

        self.fake_image_train_tf, _ = \
            self.decoder.build(self.z_pl, train=True)
        self.fake_image_test_tf, _ = \
            self.decoder.build(self.z_pl, train=False)
        if self.output_reverse:
            self.fake_image_test_tf = 1.0 - self.fake_image_test_tf

        self.fake_image_cr_train_tf_l = [
            self.decoder.build(
                self.z_cr_pl_l[0], train=self.train_with_crd_pl)[0],
            self.decoder.build(
                self.z_cr_pl_l[1], train=self.train_with_crd_pl)[0]]
        self.fake_image_cr_train_tf = \
            tf.concat(self.fake_image_cr_train_tf_l, axis=3)

        self.xd_real_image_train_tf, self.q_real_image_train_tf, _ = \
            self.infoGANDiscriminator.build(
                self.real_image_pl, train=True, sn_op=self.SN_OP)
        _, self.q_real_image_test_tf, _ = \
            self.infoGANDiscriminator.build(
                self.real_image_pl, train=False, sn_op=NO_OPS)
        self.xd_fake_image_train_tf, self.q_fake_image_train_tf, _ = \
            self.infoGANDiscriminator.build(
                self.fake_image_train_tf, train=True, sn_op=NO_OPS)

        self.reg_log_probs = self.latent.reg_log_probs(
            self.q_fake_image_train_tf, self.z_pl)
        self.reg_log_prob = self.latent.reg_log_prob(
            self.q_fake_image_train_tf, self.z_pl)
        self.reg_errors = self.latent.reg_errors(
            self.q_fake_image_train_tf, self.z_pl)

        self.crd_fake_image_train_tf, _ = \
            self.crDiscriminator.build(
                self.fake_image_cr_train_tf,
                train=True)

        self.decoder.print_layers()
        self.infoGANDiscriminator.print_layers()
        self.crDiscriminator.print_layers()

    def build_loss(self):
        self.crd_groundtruth_label_pl = tf.placeholder(
            tf.int32, [None], name="crd_groundtruth_label")
        self.crd_groundtruth = tf.one_hot(
            indices=self.crd_groundtruth_label_pl,
            depth=self.latent.num_reg_latent)
        self.cr_coe_pl = tf.placeholder(tf.float32, (), name="cr_coe")

        self.crd_acc = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(self.crd_fake_image_train_tf, axis=1),
                    tf.argmax(self.crd_groundtruth, axis=1)),
                tf.float32))

        self.de_loss_fake = \
            -tf.reduce_mean(tf.log(self.xd_fake_image_train_tf + self.EPS))
        self.de_loss_q = -tf.reduce_mean(self.reg_log_prob)
        self.de_loss_cr = tf.losses.softmax_cross_entropy(
            onehot_labels=self.crd_groundtruth,
            logits=self.crd_fake_image_train_tf)
        self.de_loss = (self.de_loss_fake +
                        self.info_coe_de * self.de_loss_q +
                        self.cr_coe_pl * self.de_loss_cr)

        self.xd_loss_real = \
            -tf.reduce_mean(tf.log(self.xd_real_image_train_tf + self.EPS))
        self.xd_loss_fake = \
            -tf.reduce_mean(tf.log(1 - self.xd_fake_image_train_tf + self.EPS))
        self.xd_loss = self.xd_loss_real + self.xd_loss_fake

        self.q_loss = -tf.reduce_mean(self.reg_log_prob)

        self.infod_loss = (self.xd_loss +
                           self.info_coe_infod * self.q_loss)

        self.crd_loss = tf.losses.softmax_cross_entropy(
            onehot_labels=self.crd_groundtruth,
            logits=self.crd_fake_image_train_tf)

        self.de_op = \
            tf.train.AdamOptimizer(self.de_lr, beta1=self.de_beta1)\
            .minimize(
                self.de_loss,
                var_list=self.decoder.trainable_vars)
        self.infod_op = \
            tf.train.AdamOptimizer(self.infod_lr, beta1=self.infod_beta1)\
            .minimize(
                self.infod_loss,
                var_list=self.infoGANDiscriminator.trainable_vars)
        with tf.variable_scope("crd_op"):
            self.crd_op = \
                tf.train.AdamOptimizer(self.crd_lr, beta1=self.crd_beta1)\
                .minimize(
                    self.crd_loss,
                    var_list=self.crDiscriminator.trainable_vars)
        self.crd_op_parameters = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, "crd_op")
        self.spectral_norm_update_ops = tf.get_collection(self.SN_OP)

    def build_summary(self):
        self.de_summary = []
        self.de_summary.append(tf.summary.scalar(
            "loss/de/fake", self.de_loss_fake))
        self.de_summary.append(tf.summary.scalar(
            "loss/de/q", self.de_loss_q))
        self.de_summary.append(tf.summary.scalar(
            "loss/de/cr", self.de_loss_cr))
        self.de_summary.append(tf.summary.scalar(
            "loss/de", self.de_loss))
        self.de_summary = tf.summary.merge(self.de_summary)

        self.infod_summary = []
        self.infod_summary.append(tf.summary.scalar(
            "loss/infod/xd/real", self.xd_loss_real))
        self.infod_summary.append(tf.summary.scalar(
            "loss/infod/xd/fake", self.xd_loss_fake))
        self.infod_summary.append(tf.summary.scalar(
            "loss/infod/xd", self.xd_loss))
        self.infod_summary.append(tf.summary.scalar(
            "loss/infod/q", self.q_loss))
        self.infod_summary.append(tf.summary.scalar(
            "loss/infod", self.infod_loss))
        self.infod_summary.append(tf.summary.scalar(
            "xd/real", tf.reduce_mean(self.xd_real_image_train_tf)))
        self.infod_summary.append(tf.summary.scalar(
            "xd/fake", tf.reduce_mean(self.xd_fake_image_train_tf)))
        for i in range(self.latent.num_reg_latent):
            self.infod_summary.append(tf.summary.scalar(
                "q/log_prob{}".format(i),
                tf.reduce_mean(self.reg_log_probs[i])))
            self.infod_summary.append(tf.summary.scalar(
                "q/mean_squared_error{}".format(i),
                tf.reduce_mean(self.reg_errors[i])))
        self.infod_summary = tf.summary.merge(self.infod_summary)

        self.crd_summary = []
        self.crd_summary.append(tf.summary.scalar(
            "loss/cr", self.crd_loss))
        self.crd_summary.append(tf.summary.scalar(
            "crd/acc", self.crd_acc))
        self.crd_summary = tf.summary.merge(self.crd_summary)

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
        # In cases where people move the checkpoint directory to another place,
        # model path indicated by get_checkpoint_state will be wrong. So we
        # get the model name and then recontruct path using checkpoint_dir
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))

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
                self.fake_image_test_tf,
                feed_dict={self.z_pl: z[i * self.batch_size:
                                        (i + 1) * self.batch_size]})
            samples.append(sub_samples)
        return np.vstack(samples)

    def inference_from(self, img):
        latents = []
        for i in range(int(math.ceil(float(img.shape[0]) / self.batch_size))):
            sub_latents = self.sess.run(
                self.q_real_image_test_tf,
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
        gap_cnt = 0
        gap = self.gap_start

        cr_coe = self.cr_coe_start
        cr_coe_cnt = 0

        data_id = np.arange(self.data.shape[0])

        for epoch_id in tqdm(range(self.epoch)):
            np.random.shuffle(data_id)

            with open(self.time_path, "a") as f:
                time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                f.write("epoch {} starts: {}\n".format(epoch_id, time))

            for batch_id in range(batch_num):
                if (global_id > 0 and
                    global_id % self.gap_decrease_batch == 0 and
                        gap_cnt < self.gap_decrease_times):

                    self.sess.run(tf.initialize_variables(
                        self.crDiscriminator.trainable_vars))
                    self.sess.run(tf.initialize_variables(
                        self.crd_op_parameters))

                    gap -= self.gap_decrease
                    gap = max(gap, 0.0)
                    gap_cnt += 1

                if (global_id > 0 and
                    global_id % self.cr_coe_increase_batch == 0 and
                        cr_coe_cnt < self.cr_coe_increase_times):

                    if cr_coe_cnt == 0:
                        self.visualize(-1, -1, global_id - 1)
                        self.log_metric(-1, -1, global_id - 1)
                        saver = tf.train.Saver()
                        checkpoint_dir = os.path.join(
                            self.checkpoint_dir,
                            "global_id-{}".format(global_id - 1))
                        self.save(global_id - 1, saver, checkpoint_dir)

                    self.sess.run(tf.initialize_variables(
                        self.crDiscriminator.trainable_vars))
                    self.sess.run(tf.initialize_variables(
                        self.crd_op_parameters))

                    cr_coe += self.cr_coe_increase
                    cr_coe_cnt += 1

                batch_data_id = \
                    data_id[batch_id * self.batch_size:
                            (batch_id + 1) * self.batch_size]
                batch_image = self.data[batch_data_id]
                batch_z = self.latent.sample(self.batch_size)

                crd_groundtruth_label = np.random.choice(
                    self.latent.num_reg_latent, self.batch_size)

                alpha_same = np.zeros((self.batch_size, 1))
                batch_z_cr = self.latent.sample_cr(
                    self.batch_size, 2,
                    alpha_same,
                    alpha_same + gap,
                    crd_groundtruth_label)
                feed_dict = {
                    self.cr_coe_pl: cr_coe,
                    self.z_pl: batch_z,
                    self.z_cr_pl_l[0]: batch_z_cr[0],
                    self.z_cr_pl_l[1]: batch_z_cr[1],
                    self.real_image_pl: batch_image,
                    self.crd_groundtruth_label_pl: crd_groundtruth_label,
                    self.train_with_crd_pl: cr_coe > self.EPS}

                summary_result, _ = self.sess.run(
                    [self.crd_summary, self.crd_op],
                    feed_dict=feed_dict)
                if global_id % self.summary_freq == 0:
                    self.summary_writer.add_summary(summary_result, global_id)

                summary_result, _ = self.sess.run(
                    [self.infod_summary, self.infod_op],
                    feed_dict=feed_dict)
                if global_id % self.summary_freq == 0:
                    self.summary_writer.add_summary(summary_result, global_id)

                _ = self.sess.run(
                    [self.de_op],
                    feed_dict=feed_dict)

                summary_result, _ = self.sess.run(
                    [self.de_summary, self.de_op],
                    feed_dict=feed_dict)
                if global_id % self.summary_freq == 0:
                    self.summary_writer.add_summary(summary_result, global_id)

                if global_id % self.summary_freq == 0:
                    summary = tf.Summary(
                        value=[tf.Summary.Value(
                            tag="train/gap", simple_value=gap)])
                    self.summary_writer.add_summary(summary, global_id)
                    summary = tf.Summary(
                        value=[tf.Summary.Value(
                            tag="train/cr_coe", simple_value=cr_coe)])
                    self.summary_writer.add_summary(summary, global_id)

                self.sess.run(self.spectral_norm_update_ops)

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
    from latent import UniformLatent, JointLatent
    from network import Decoder, InfoGANDiscriminator, CrDiscriminator
    from load_data import load_3Dpots
    from metric import FactorVAEMetric

    data, metric_data, latent_values = \
        load_3Dpots("../data/3Dpots")
    _, height, width, depth = data.shape
    print(data.min(), data.max())

    latent_list = []
    for i in range(5):
        latent_list.append(
            UniformLatent(
                in_dim=1, out_dim=1, low=-1.0, high=1.0, apply_reg=True))
    latent_list.append(
        UniformLatent(
            in_dim=5, out_dim=5, low=-1.0, high=1.0, apply_reg=False))
    latent = JointLatent(latent_list=latent_list)

    decoder = Decoder(
        output_width=width, output_height=height, output_depth=depth)
    infoGANDiscriminator = \
        InfoGANDiscriminator(output_length=latent.reg_out_dim)
    crDiscriminator = \
        CrDiscriminator(output_length=latent.num_reg_latent)

    checkpoint_dir = "./test/checkpoint"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    sample_dir = "./test/sample"
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    time_path = "./test/time.txt"
    metric_path = "./test/metric.csv"
    epoch = 40
    batch_size = 64
    vis_freq = 200
    vis_num_sample = 10
    vis_num_rep = 10
    metric_freq = 400
    info_coe_de = 0.2
    info_coe_infod = 0.2
    gap_start = 1.9
    gap_decrease_times = 1
    gap_decrease = 1.9
    gap_decrease_batch = 85000
    cr_coe_start = 0.0
    cr_coe_increase_times = 1
    cr_coe_increase = 3.0
    cr_coe_increase_batch = 50000

    run_config = tf.ConfigProto()
    with tf.Session(config=run_config) as sess:
        factorVAEMetric = FactorVAEMetric(metric_data, sess=sess)
        gan = INFOGAN_CR(
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
            infoGANDiscriminator=infoGANDiscriminator,
            crDiscriminator=crDiscriminator,
            gap_start=gap_start,
            gap_decrease_times=gap_decrease_times,
            gap_decrease=gap_decrease,
            gap_decrease_batch=gap_decrease_batch,
            cr_coe_start=cr_coe_start,
            cr_coe_increase_times=cr_coe_increase_times,
            cr_coe_increase=cr_coe_increase,
            cr_coe_increase_batch=cr_coe_increase_batch,
            info_coe_de=info_coe_de,
            info_coe_infod=info_coe_infod,
            metric_callbacks=[factorVAEMetric],
            metric_freq=metric_freq,
            metric_path=metric_path,
            output_reverse=False)
        gan.build()
        gan.train()
