import numpy as np
import tensorflow as tf
import math


class Latent(object):
    def __init__(self, apply_reg=True):
        self.apply_reg = apply_reg

    @property
    def in_dim(self):
        return self._in_dim

    @property
    def out_dim(self):
        return self._out_dim

    @property
    def reg_in_dim(self):
        return self._in_dim if self.apply_reg else 0

    @property
    def reg_out_dim(self):
        return self._out_dim if self.apply_reg else 0

    @property
    def num_latent(self):
        return 1

    @property
    def num_reg_latent(self):
        return 1 if self.apply_reg else 0

    def sample(self, batch_size):
        raise NotImplementedError

    def sample_same(self, batch_size, group_num):
        raise NotImplementedError

    def sample_different(self, batch_size, group_num):
        raise NotImplementedError

    def uniformly_sample(self, num_sample, num_rep):
        raise NotImplementedError

    def shuffled_reg_out_tf_var(self, out_tf_var):
        return tf.random_shuffle(out_tf_var) if self.apply_reg else None

    def reg_log_prob(self, out_tf_var, ground_truth):
        raise NotImplementedError

    def reg_error(self, out_tf_var, ground_truth):
        raise NotImplementedError


class UniformLatent(Latent):
    def __init__(self, in_dim, out_dim, low=-1.0, high=1.0, q_std=1.0,
                 *args, **kwargs):
        super(UniformLatent, self).__init__(*args, **kwargs)
        self.low = low
        self.high = high
        self.q_std = q_std
        self._in_dim = in_dim
        self._out_dim = out_dim

    def sample(self, batch_size):
        return self._sample(batch_size, self.low, self.high)

    def _sample(self, batch_size, low, high):
        return np.random.uniform(
            low, high, size=(batch_size, self._in_dim))

    def _sample_alpha(self, low, high):
        data = []
        for i in range(low.shape[0]):
            data.append(self._sample(1, low[i], high[i]))
        return np.vstack(data)

    def sample_same(self, batch_size, group_num, alpha):
        assert group_num == 2
        if isinstance(alpha, (int, long, float, complex)):
            alpha = np.ones(shape=(batch_size, 1)) * alpha
        assert np.all(alpha <= self.high - self.low)
        data = self._sample_alpha(self.low + alpha / 2, self.high - alpha / 2)
        sign = np.random.choice([-1.0, 1.0], size=data.shape)
        return [data + sign * alpha / 2, data - sign * alpha / 2]

    def sample_different(self, batch_size, group_num, alpha):
        assert group_num == 2
        if isinstance(alpha, (int, long, float, complex)):
            alpha = np.ones(shape=(batch_size, 1)) * alpha
        assert np.all(alpha <= self.high - self.low)
        data = []
        for _ in range(group_num):
            data.append(self._sample_alpha(
                self.low + alpha / 2, self.high - alpha / 2))
        sign = (data[0] > data[1]) * 2.0 - 1.0
        data[0] = data[0] + sign * alpha / 2
        data[1] = data[1] - sign * alpha / 2
        return data

    def uniformly_sample(self, num_sample, num_rep):
        data = np.stack(
            [np.linspace(self.low, self.high, num_sample)] * self._in_dim,
            axis=1)
        return np.repeat(data, num_rep, axis=0)

    def reg_log_prob(self, out_tf_var, ground_truth):
        prob = (-tf.square(out_tf_var - ground_truth)
                / 2. / (self.q_std**2.))
        prob += -math.log(2. * math.pi) / 2. - math.log(self.q_std)
        return tf.reduce_sum(prob, axis=1, keep_dims=True)

    def reg_error(self, out_tf_var, ground_truth):
        error = tf.square(out_tf_var - ground_truth)
        return tf.reduce_sum(error, axis=1, keep_dims=True)


class GaussianLatent(Latent):
    def __init__(self, in_dim, out_dim, loc=0.0, scale=1.0, q_std=1.0,
                 *args, **kwargs):
        super(GaussianLatent, self).__init__(*args, **kwargs)
        self.loc = loc
        self.scale = scale
        self.q_std = q_std
        self._in_dim = in_dim
        self._out_dim = out_dim

    def sample(self, batch_size):
        return np.random.normal(
            loc=self.loc, scale=self.scale, size=(batch_size, self._in_dim))

    def sample_same(self, batch_size, group_num):
        data = self.sample(batch_size)
        return [data] * group_num

    def sample_different(self, batch_size, group_num):
        data = []
        for _ in range(group_num):
            data.append(self.sample(batch_size))
        return data

    def uniformly_sample(self, num_sample, num_rep, low=None, high=None):
        if low is None or high is None:
            low = self.loc - 2 * self.scale
            high = self.loc + 2 * self.scale
        data = np.stack(
            [np.linspace(low, high, num_sample)] * self._in_dim,
            axis=1)
        return np.repeat(data, num_rep, axis=0)

    def reg_log_prob(self, out_tf_var, ground_truth):
        prob = (-tf.square(out_tf_var - ground_truth)
                / 2. / (self.q_std**2.))
        prob += -math.log(2. * math.pi) / 2. - math.log(self.q_std)
        return tf.reduce_sum(prob, axis=1, keep_dims=True)

    def reg_error(self, out_tf_var, ground_truth):
        error = tf.square(out_tf_var - ground_truth)
        return tf.reduce_sum(error, axis=1, keep_dims=True)


class JointLatent(Latent):
    def __init__(self, latent_list, *args, **kwargs):
        # JointLatent's apply_tc is ignored
        super(JointLatent, self).__init__(*args, **kwargs)
        self.latent_list = latent_list
        self._in_dims = [latent.in_dim for latent in latent_list]
        self._in_dim = sum(self._in_dims)
        self._out_dims = [latent.out_dim for latent in latent_list]
        self._out_dim = sum(self._out_dims)
        self._reg_latent_list = [latent
                                 for latent in self.latent_list
                                 if latent.apply_reg]
        self._reg_latent_ids = [i
                                for i, latent in enumerate(self.latent_list)
                                if latent.apply_reg]
        self._reg_in_dims = [latent.reg_in_dim
                             for latent in self._reg_latent_list]
        self._reg_in_dim = np.sum(self._reg_in_dims)
        self._reg_out_dims = [latent.reg_out_dim
                              for latent in self._reg_latent_list]
        self._reg_out_dim = np.sum(self._reg_out_dims)
        self._num_latent = np.sum(
            [latent.num_latent for latent in self.latent_list])
        self._num_reg_latent = np.sum(
            [latent.num_reg_latent
             for latent in self.latent_list if latent.apply_reg])

    @property
    def reg_latent_list(self):
        return self._reg_latent_list

    @property
    def reg_in_dim(self):
        return self._reg_in_dim

    @property
    def reg_out_dim(self):
        return self._reg_out_dim

    @property
    def reg_in_dims(self):
        return self._reg_in_dims

    @property
    def reg_out_dims(self):
        return self._reg_out_dims

    @property
    def num_latent(self):
        return self._num_latent

    @property
    def num_reg_latent(self):
        return self._num_reg_latent

    @property
    def reg_latent_ids(self):
        return self._reg_latent_ids

    def sample(self, batch_size, latent_id=None):
        if latent_id is None:
            latent_id = list(range(self.num_latent))
        samples = [latent.sample(batch_size)
                   for i, latent in enumerate(self.latent_list)
                   if i in latent_id]
        return np.concatenate(samples, axis=1)

    def sample_cr(self, batch_size, group_num, alpha_same, alpha_different,
                  select_id):
        samples = [np.zeros((batch_size, self._in_dim))
                   for _ in range(group_num)]
        col_id = 0
        reg_id = 0
        for i, latent_i in enumerate(self.latent_list):
            if not latent_i.apply_reg:  # z
                sub_samples = latent_i.sample_same(batch_size, group_num, 0.0)
            else:  # q
                sub_samples = [np.zeros((batch_size, latent_i.in_dim))
                               for _ in range(group_num)]

                rows_select = np.where(select_id == reg_id)[0]
                if rows_select.shape[0] > 0:
                    sub_samples_select = latent_i.sample_same(
                        rows_select.shape[0], group_num,
                        alpha_same[rows_select])
                    for j in range(group_num):
                        sub_samples[j][rows_select, :] = sub_samples_select[j]

                rows_not_select = np.where(select_id != reg_id)[0]
                if rows_not_select.shape[0] > 0:
                    sub_samples_not_select = latent_i.sample_different(
                        rows_not_select.shape[0], group_num,
                        alpha_different[rows_not_select])
                    for j in range(group_num):
                        sub_samples[j][rows_not_select, :] = \
                            sub_samples_not_select[j]

                reg_id += 1

            for j in range(group_num):
                samples[j][:, col_id: col_id + latent_i.in_dim] = \
                    sub_samples[j]
            col_id += latent_i.in_dim

        return samples

    def uniformly_sample(self, num_sample, num_rep):
        base_data = self.sample(num_rep)
        base_data = np.tile(base_data, (num_sample, 1))

        samples = []

        col = 0
        for latent in self.latent_list:
            data = np.copy(base_data)
            sub_data = latent.uniformly_sample(num_sample, num_rep)
            data[:, col: col + latent.in_dim] = sub_data
            samples.append(data)
            col += latent.in_dim

        return samples

    def shuffled_reg_out_tf_var(self, out_tf_var):
        out_tf_var = tf.split(out_tf_var, self.reg_out_dims, axis=1)
        reg_out_tf_vars = []
        for i, latent in enumerate(self._reg_latent_list):
            reg_out_tf_vars.append(
                latent.shuffled_reg_out_tf_var(out_tf_var[i]))
        return tf.concat(reg_out_tf_vars, axis=1)

    def reg_log_probs(self, out_tf_var, ground_truth):
        prob = []

        ground_truth = tf.split(ground_truth, self._out_dims, axis=1)
        out_tf_var = tf.split(out_tf_var, self._reg_out_dims, axis=1)

        reg_i = 0
        for i, latent in enumerate(self.latent_list):
            if latent.apply_reg:
                prob.append(
                    latent.reg_log_prob(out_tf_var[reg_i], ground_truth[i]))
                reg_i += 1
        return prob

    def reg_errors(self, out_tf_var, ground_truth):
        prob = []

        ground_truth = tf.split(ground_truth, self._out_dims, axis=1)
        out_tf_var = tf.split(out_tf_var, self._reg_out_dims, axis=1)

        reg_i = 0
        for i, latent in enumerate(self.latent_list):
            if latent.apply_reg:
                prob.append(
                    latent.reg_error(out_tf_var[reg_i], ground_truth[i]))
                reg_i += 1
        return prob

    def reg_log_prob(self, out_tf_var, ground_truth):
        prob = self.reg_log_probs(out_tf_var, ground_truth)
        prob = tf.concat(prob, axis=1)
        return tf.reduce_sum(prob, axis=1, keep_dims=True)


if __name__ == "__main__":
    latent_list = []
    for i in range(2):
        latent_list.append(
            UniformLatent(in_dim=1, out_dim=1, low=-1.0, high=1.0,
                          apply_reg=True))
    latent = JointLatent(latent_list=latent_list)
    print(latent.sample(10))
