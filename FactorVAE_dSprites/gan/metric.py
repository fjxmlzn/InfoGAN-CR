import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
import math
from collections import Counter
import scipy.stats
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
from sklearn.svm import LinearSVC
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


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
    """ Impementation of the metric in: 
        Disentangling by Factorising
    """
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


class BetaVAEMetric(Metric):
    """ Impementation of the metric in: 
        beta-VAE: Learning Basic Visual Concepts with a Constrained Variational 
        Framework
    """
    def __init__(self, metric_data, *args, **kwargs):
        super(BetaVAEMetric, self).__init__(*args, **kwargs)
        self.metric_data = metric_data

    def evaluate(self, epoch_id, batch_id, global_id):
        features = []
        labels = []

        for data in self.metric_data["groups"]:
            data_inference = self.model.inference_from(data["img"])
            data_diff = np.abs(data_inference[0::2] - data_inference[1::2])
            data_diff_mean = np.mean(data_diff, axis=0)
            features.append(data_diff_mean)
            labels.append(data["label"])

        features = np.vstack(features)
        labels = np.asarray(labels)

        classifier =  LogisticRegression()
        classifier.fit(features, labels)

        acc = classifier.score(features, labels)

        return {"betaVAE_metric": acc}


class SAPMetric(Metric):
    """ Impementation of the metric in: 
        VARIATIONAL INFERENCE OF DISENTANGLED LATENT CONCEPTS FROM UNLABELED 
        OBSERVATIONS
        Part of the code is adapted from:
        https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/sap_score.py
    """
    def __init__(self, metric_data, *args, **kwargs):
        super(SAPMetric, self).__init__(*args, **kwargs)
        self.metric_data = metric_data

    def evaluate(self, epoch_id, batch_id, global_id):
        data_inference = self.model.inference_from(
            self.metric_data["img_with_latent"]["img"])
        data_gt_latents = self.metric_data["img_with_latent"]["latent"]
        factor_is_continuous = \
            self.metric_data["img_with_latent"]["is_continuous"]

        num_latents = data_inference.shape[1]
        num_factors = len(factor_is_continuous)

        score_matrix = np.zeros([num_latents, num_factors])
        for i in range(num_latents):
            for j in range(num_factors):
                inference_values = data_inference[:, i]
                gt_values = data_gt_latents[:, j]
                if factor_is_continuous[j]:
                    cov = np.cov(inference_values, gt_values, ddof=1)
                    assert np.all(np.asarray(list(cov.shape)) == 2)
                    cov_cov = cov[0, 1]**2
                    cov_sigmas_1 = cov[0, 0]
                    cov_sigmas_2 = cov[1, 1]
                    score_matrix[i, j] = cov_cov / cov_sigmas_1 / cov_sigmas_2
                else:
                    gt_values = gt_values.astype(np.int32)
                    classifier = LinearSVC(C=0.01, class_weight="balanced")
                    classifier.fit(inference_values[:, np.newaxis], gt_values)
                    pred = classifier.predict(inference_values[:, np.newaxis])
                    score_matrix[i, j] = np.mean(pred == gt_values)
        sorted_score_matrix = np.sort(score_matrix, axis=0)
        score = np.mean(sorted_score_matrix[-1, :] - 
                        sorted_score_matrix[-2, :])

        return {"SAP_metric": score,
                "SAP_metric_detail": score_matrix}


class MIGMetric(Metric):
    """ Impementation of the metric in: 
        Isolating Sources of Disentanglement in Variational Autoencoders
        Part of the code is adapted from:
        https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/mig.py
    """
    def __init__(self, metric_data, *args, **kwargs):
        super(MIGMetric, self).__init__(*args, **kwargs)
        self.metric_data = metric_data

    def discretize(self, data, num_bins=20):
        """ Adapted from:
            https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/utils.py
        """
        discretized = np.zeros_like(data)
        for i in range(data.shape[1]):
            discretized[:, i] = np.digitize(
                data[:, i],
                np.histogram(data[:, i], num_bins)[1][:-1])
        return discretized

    def mutual_info(self, data1, data2):
        """ Adapted from:
            https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/utils.py
        """
        n1 = data1.shape[1]
        n2 = data2.shape[1]
        mi = np.zeros([n1, n2])
        for i in range(n1):
            for j in range(n2):
                mi[i, j] = mutual_info_score(
                    data2[:, j], data1[:, i])
        return mi

    def entropy(self, data):
        """ Adapted from:
            https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/utils.py
        """
        num_factors = data.shape[1]
        entr = np.zeros(num_factors)
        for i in range(num_factors):
            entr[i] = mutual_info_score(data[:, i], data[:, i])
        return entr

    def evaluate(self, epoch_id, batch_id, global_id):
        data_inference = self.model.inference_from(
            self.metric_data["img_with_latent"]["img"])
        data_gt_latents = self.metric_data["img_with_latent"]["latent_id"]

        data_inference_discrete = self.discretize(data_inference)
        mi = self.mutual_info(
            data_inference_discrete, data_gt_latents)
        entropy = self.entropy(data_gt_latents)
        sorted_mi = np.sort(mi, axis=0)[::-1]
        mig_score = np.mean(
            np.divide(sorted_mi[0, :] - sorted_mi[1, :], entropy))

        return {"MIG_metric": mig_score,
                "MIG_metric_detail_mi": mi,
                "MIG_metric_detail_entropy": entropy}


class FStatMetric(Metric):
    """ Impementation of the metric in: 
        Learning Deep Disentangled Embeddings With the F-Statistic Loss
        Part of the code is adapted from:
        https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/modularity_explicitness.py
    """
    def __init__(self, metric_data, *args, **kwargs):
        super(FStatMetric, self).__init__(*args, **kwargs)
        self.metric_data = metric_data

    def discretize(self, data, num_bins=20):
        """ Adapted from:
            https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/utils.py
        """
        discretized = np.zeros_like(data)
        for i in range(data.shape[1]):
            discretized[:, i] = np.digitize(
                data[:, i],
                np.histogram(data[:, i], num_bins)[1][:-1])
        return discretized

    def mutual_info(self, data1, data2):
        """ Adapted from:
            https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/utils.py
        """
        n1 = data1.shape[1]
        n2 = data2.shape[1]
        mi = np.zeros([n1, n2])
        for i in range(n1):
            for j in range(n2):
                mi[i, j] = mutual_info_score(
                    data2[:, j], data1[:, i])
        return mi

    def evaluate(self, epoch_id, batch_id, global_id):
        data_inference = self.model.inference_from(
            self.metric_data["img_with_latent"]["img"])
        data_gt_latent_ids = self.metric_data["img_with_latent"]["latent_id"]

        data_inference_discrete = self.discretize(data_inference)
        modu_mi = self.mutual_info(
            data_inference_discrete, data_gt_latent_ids)
        squared_modu_mi = np.square(modu_mi)
        max_squared_modu_mi = np.max(squared_modu_mi, axis=1)
        numerator = np.sum(squared_modu_mi, axis=1) - max_squared_modu_mi
        denominator = max_squared_modu_mi * (data_gt_latent_ids.shape[1] - 1)
        modu_delta = numerator / denominator
        modu_score_detail = 1.0 - modu_delta
        modu_score = np.mean(modu_score_detail)

        expl_score_detail = np.zeros([data_gt_latent_ids.shape[1], 1])
        for i in range(data_gt_latent_ids.shape[1]):
            classifier = LogisticRegression()
            y = data_gt_latent_ids[:, i]
            classifier.fit(data_inference, y)
            pred_brob = classifier.predict_proba(data_inference)
            mlb = MultiLabelBinarizer()
            roc = roc_auc_score(
                mlb.fit_transform(np.expand_dims(y, 1)),
                pred_brob)
            expl_score_detail[i] = roc
        expl_score = np.mean(expl_score_detail)

        return {"FStat_modu_metric": modu_score,
                "FStat_modu_metric_detail": modu_score_detail,
                "FStat_modu_mi": modu_mi,
                "FStat_expl_metric": expl_score,
                "FStat_expl_metric_detail": expl_score_detail}


class DCIMetric(Metric):
    """ Impementation of the metric in: 
        A FRAMEWORK FOR THE QUANTITATIVE EVALUATION OF DISENTANGLED 
        REPRESENTATIONS
        Part of the code is adapted from:
        https://github.com/cianeastwood/qedr
    """
    def __init__(self, metric_data, regressor="Lasso", *args, **kwargs):
        super(DCIMetric, self).__init__(*args, **kwargs)
        self.data = metric_data["img_with_latent"]["img"]
        self.latents = metric_data["img_with_latent"]["latent"]

        self._regressor = regressor
        if regressor == "Lasso":
            self.regressor_class = Lasso
            self.alpha = 0.02
            # constant alpha for all models and targets
            self.params = {"alpha": self.alpha} 
            # weights
            self.importances_attr = "coef_" 
        elif regressor == "LassoCV":
            self.regressor_class = LassoCV
            # constant alpha for all models and targets
            self.params = {} 
            # weights
            self.importances_attr = "coef_" 
        elif regressor == "RandomForest":
            self.regressor_class = RandomForestRegressor
            # Create the parameter grid based on the results of random search 
            max_depths = [4, 5, 2, 5, 5]
            # Create the parameter grid based on the results of random search 
            self.params = [{"max_depth": max_depth, "oob_score": True}
                           for max_depth in max_depths]
            self.importances_attr = "feature_importances_"
        elif regressor == "RandomForestIBGAN":
            # The parameters that IBGAN paper uses
            self.regressor_class = RandomForestRegressor
            # Create the parameter grid based on the results of random search 
            max_depths = [4, 2, 4, 2, 2]
            # Create the parameter grid based on the results of random search 
            self.params = [{"max_depth": max_depth, "oob_score": True}
                           for max_depth in max_depths]
            self.importances_attr = "feature_importances_"
        elif regressor == "RandomForestCV":
            self.regressor_class = GridSearchCV
            # Create the parameter grid based on the results of random search 
            param_grid = {"max_depth": [i for i in range(2, 16)]}
            self.params = {
                "estimator": RandomForestRegressor(),
                "param_grid": param_grid,
                "cv": 3,
                "n_jobs": -1,
                "verbose": 0
            }
            self.importances_attr = "feature_importances_"
        elif "RandomForestEnum" in regressor:
            self.regressor_class = RandomForestRegressor
            # Create the parameter grid based on the results of random search 
            self.params = {
                "max_depth": int(regressor[len("RandomForestEnum"):]),
                "oob_score": True
            }
            self.importances_attr = "feature_importances_"
        else:
            raise NotImplementedError()

        self.TINY = 1e-12

    def normalize(self, X):
        mean = np.mean(X, 0) # training set
        stddev = np.std(X, 0) # training set
        #print('mean', mean)
        #print('std', stddev)
        return (X - mean) / stddev

    def norm_entropy(self, p):
        '''p: probabilities '''
        n = p.shape[0]
        return - p.dot(np.log(p + self.TINY) / np.log(n + self.TINY))

    def entropic_scores(self, r):
        '''r: relative importances '''
        r = np.abs(r)
        ps = r / np.sum(r, axis=0) # 'probabilities'
        hs = [1 - self.norm_entropy(p) for p in ps.T]
        return hs

    def evaluate(self, epoch_id, batch_id, global_id):
        codes = self.model.inference_from(self.data)
        latents = self.latents
        codes = self.normalize(codes)
        latents = self.normalize(latents)
        R = []

        for j in range(self.latents.shape[-1]):
            if isinstance(self.params, dict):
              regressor = self.regressor_class(**self.params)
            elif isinstance(self.params, list):
              regressor = self.regressor_class(**self.params[j])
            regressor.fit(codes, latents[:, j])

            # extract relative importance of each code variable in 
            # predicting the latent z_j
            if self._regressor == "RandomForestCV":
                best_rf = regressor.best_estimator_
                r = getattr(best_rf, self.importances_attr)[:, None]
            else:
                r = getattr(regressor, self.importances_attr)[:, None]

            R.append(np.abs(r))

        R = np.hstack(R) #columnwise, predictions of each z

        # disentanglement
        disent_scores = self.entropic_scores(R.T)
        # relative importance of each code variable
        c_rel_importance = np.sum(R, 1) / np.sum(R) 
        disent_w_avg = np.sum(np.array(disent_scores) * c_rel_importance)

        # completeness
        complete_scores = self.entropic_scores(R)
        complete_avg = np.mean(complete_scores)

        return {
            "DCI_{}_disent_metric_detail".format(self._regressor): \
                disent_scores,
            "DCI_{}_disent_metric".format(self._regressor): disent_w_avg,
            "DCI_{}_complete_metric_detail".format(self._regressor): \
                complete_scores,
            "DCI_{}_complete_metric".format(self._regressor): complete_avg,
            "DCI_{}_metric_detail".format(self._regressor): R
            }
