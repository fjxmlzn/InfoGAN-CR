import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
import math
from collections import Counter
import scipy.stats


class Metric(object):
    def __init__(self, sess):
        self.sess = sess
        self.model = None

    def set_model(self, model):
        self.model = model

    def build(self):
        pass

    def load(self):
        pass

    def evaluate(self, epoch_id, batch_id, global_id):
        raise NotImplementedError


class DSpritesInceptionScore(Metric):
    def __init__(self, do_training, data, metadata, latent_values,
                 network_path,
                 shape_network,
                 lr=0.01, epoch=5, batch_size=64, checkpoint_dir=None,
                 evaluate_num_samples=50000,
                 sample_dir=None,
                 *args, **kwargs):
        super(DSpritesInceptionScore, self).__init__(*args, **kwargs)
        self.do_training = do_training
        self.data = data
        self.metadata = metadata
        self.latent_values = latent_values
        self.network_path = network_path
        self.shape_network = shape_network
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.evaluate_num_samples = evaluate_num_samples
        self.sample_dir = sample_dir

        self.image_dims = list(data.shape[1:])

        self.shape_network_path = os.path.join(self.network_path, "shape")

        self.EPS = 1e-8

    def build(self):
        self.build_connection()
        if self.do_training:
            self.build_label()
            self.build_loss()
            self.build_summary()

    def build_label(self):
        self.shape_groundtruth = \
            np.eye(3)[self.latent_values[:, 1].astype(np.int32) - 1]

    def build_loss(self):
        self.shape_groundtruth_pl = tf.placeholder(
            tf.float32,
            [None, 3],
            name="dSpritesSampleQualityMetric_shape_groundtruth")

        self.shape_loss = tf.losses.softmax_cross_entropy(
            onehot_labels=self.shape_groundtruth_pl,
            logits=self.shape_train_tf)

        self.shape_op = \
            tf.train.AdamOptimizer(self.lr).minimize(
                self.shape_loss,
                var_list=self.shape_network.trainable_vars)

    def build_summary(self):
        self.shape_summary = []
        self.shape_summary.append(tf.summary.scalar(
            "loss/shape", self.shape_loss))
        self.shape_summary = tf.summary.merge(self.shape_summary)

    def build_connection(self):
        self.generated_image_pl = tf.placeholder(
            tf.float32,
            [None] + self.image_dims,
            name="dSpritesSampleQualityMetric_generated_image")

        self.shape_train_tf, _ = \
            self.shape_network.build(self.generated_image_pl, train=True)
        self.shape_test_tf, _ = \
            self.shape_network.build(self.generated_image_pl, train=False)
        self.shape_test_tf = tf.nn.softmax(self.shape_test_tf)

        self.shape_network.print_layers()

    def train(self):
        tf.global_variables_initializer().run()
        self.summary_writer = tf.summary.FileWriter(
            self.checkpoint_dir, self.sess.graph)
        batch_num = len(self.data) // self.batch_size

        train_tasks = {
            "shape": {"label": self.shape_groundtruth,
                      "label_pl": self.shape_groundtruth_pl,
                      "op": self.shape_op,
                      "summary": self.shape_summary}}

        for task_name in train_tasks:
            print("Training {}".format(task_name))

            data_id = np.arange(self.data.shape[0])

            global_id = 0
            for epoch_id in tqdm(range(self.epoch)):
                np.random.shuffle(data_id)

                for batch_id in range(batch_num):
                    batch_data_id = \
                        data_id[batch_id * self.batch_size:
                                (batch_id + 1) * self.batch_size]
                    batch_data = self.data[batch_data_id]
                    batch_label = \
                        train_tasks[task_name]["label"][batch_data_id]

                    feed_dict = {
                        train_tasks[task_name]["label_pl"]: batch_label,
                        self.generated_image_pl: batch_data
                    }
                    summary_result, _ = self.sess.run(
                        [train_tasks[task_name]["summary"],
                         train_tasks[task_name]["op"]],
                        feed_dict=feed_dict)
                    self.summary_writer.add_summary(summary_result, global_id)

                    global_id += 1
        self.save()

    def save(self):
        self.shape_network.save(self.sess, self.shape_network_path)

    def load(self):
        self.shape_network.load(self.sess, self.shape_network_path)

    def get_network_output(self, images, tf_var):
        outputs = []
        for i in range(
                int(math.ceil(float(images.shape[0]) / self.batch_size))):
            sub_outputs = self.sess.run(
                tf_var,
                feed_dict={self.generated_image_pl:
                           images[i * self.batch_size:
                                  (i + 1) * self.batch_size]})
            outputs.append(sub_outputs)
        return np.vstack(outputs)

    def get_shape(self, images):
        return self.get_network_output(images, self.shape_test_tf)

    def _evaluate(self, shape):
        p_y = np.mean(shape, 0)
        KL = np.asarray([scipy.stats.entropy(shape[i, :], p_y)
                         for i in range(shape.shape[0])])
        is_mean = np.exp(np.mean(KL))
        is_log_std = np.std(KL)

        confidence = np.mean(np.amax(shape, axis=1))

        shape_id = np.argmax(shape, axis=1)
        counter = Counter(shape_id)
        p = np.asarray([counter[0], counter[1], counter[2]], dtype=np.float32)
        p = p / np.sum(p)
        q = np.asarray([1.0 / 3.0] * 3)
        mode_kl = scipy.stats.entropy(p, q)

        return {"DSprites_IS_mean": is_mean,
                "DSprites_IS_log_std": is_log_std,
                "DSprites_confidence": confidence,
                "DSprites_mode_KL": mode_kl,
                "DSprites_mode_detail": dict(counter)}

    def evaluate(self, epoch_id, batch_id, global_id):
        latents = self.model.latent.sample(self.evaluate_num_samples)
        images = self.model.sample_from(latents)
        shape = self.get_shape(images)
        return self._evaluate(shape)

    def evaluate_groudtruth(self):
        id_ = np.arange(self.data.shape[0])
        np.random.shuffle(id_)
        id_ = id_[0: self.evaluate_num_samples]
        images = self.data[id_]
        shape = self.get_shape(images)
        return self._evaluate(shape)


class DHSICMetric(Metric):
    def __init__(self, data, n=5000, *args, **kwargs):
        super(DHSICMetric, self).__init__(*args, **kwargs)
        self.data = data
        self.n = n

        self.eva_data_id = np.random.choice(
            np.arange(data.shape[0]), size=(self.n,), replace=False)
        self.eva_data = self.data[self.eva_data_id]

    def _get_dHSIC(self, eva_latent):
        # print(eva_latent.shape)

        eval_latent_repeat1 = np.repeat(eva_latent, self.n, axis=0)
        eval_latent_repeat2 = np.tile(eva_latent, (self.n, 1))
        eval_latent_diff = np.abs(
            eval_latent_repeat1 - eval_latent_repeat2)
        # print(eval_latent_diff.shape)

        h = np.median(eval_latent_diff, axis=0, keepdims=True)
        # print(h.shape)

        matrix = np.exp(-eval_latent_diff / np.square(h))
        matrix = np.reshape(matrix, (self.n, self.n, eva_latent.shape[1]))

        dhsic1 = np.sum(np.prod(matrix, 2) / (float(self.n)**2))
        dhsic2 = np.prod(np.sum(matrix, (0, 1)) / (float(self.n)**2))
        dhsic3 = (-2.0 *
                  np.sum(np.prod(np.sum(matrix, 1), 1) /
                         (float(self.n)**(eva_latent.shape[1] + 1))))

        dhsic = dhsic1 + dhsic2 + dhsic3

        return dhsic, dhsic1, dhsic2, dhsic3

    def evaluate(self, epoch_id, batch_id, global_id):
        eva_latent = self.model.inference_from(self.eva_data)
        eva_latent = eva_latent.astype(np.float64)

        dhsic, dhsic1, dhsic2, dhsic3 = self._get_dHSIC(eva_latent)

        return {"dHSIC": dhsic,
                "dHSIC_detail": [dhsic, dhsic1, dhsic2, dhsic3]}


class FactorVAEMetric(Metric):
    def __init__(self, metric_data, *args, **kwargs):
        super(FactorVAEMetric, self).__init__(*args, **kwargs)
        self.metric_data = metric_data

    def evaluate(self, epoch_id, batch_id, global_id):
        eval_std_inference = self.model.inference_from(
            self.metric_data["img_eval_std"])
        eval_std = np.std(eval_std_inference, axis=0, keepdims=True)

        labels = set(data["label"] for data in self.metric_data["groups"])

        train_data = np.zeros((eval_std.shape[1], len(labels)))

        for data in self.metric_data["groups"]:
            data_inference = self.model.inference_from(data["img"])
            data_inference /= eval_std
            data_std = np.std(data_inference, axis=0)
            predict = np.argmin(data_std)
            train_data[predict, data["label"]] += 1

        total_sample = np.sum(train_data)
        maxs = np.amax(train_data, axis=1)
        correct_sample = np.sum(maxs)

        correct_sample_revised = np.flip(np.sort(maxs), axis=0)
        correct_sample_revised = np.sum(
            correct_sample_revised[0: train_data.shape[1]])

        return {"factorVAE_metric": float(correct_sample) / total_sample,
                "factorVAE_metric_revised": (float(correct_sample_revised) /
                                             total_sample),
                "factorVAE_metric_detail": train_data}
