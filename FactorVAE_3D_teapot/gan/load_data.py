import os
import numpy as np


def load_3Dpots(path):
    train_data = np.load(os.path.join(path, "train_data", "data.npz"))
    metric_data = np.load(os.path.join(path, "metric_data", "data.npz"))

    imgs = train_data["imgs"]
    latent_values = train_data["latents"]

    metric_imgs = metric_data["imgs"]
    metric_labels = metric_data["labels"]

    metric_data_groups = []
    M = metric_data["imgs"].shape[0]
    for i in range(M):
        metric_data_groups.append(
            {"img": metric_imgs[i],
             "label": metric_labels[i]})

    selected_ids = np.random.permutation(range(imgs.shape[0]))
    selected_ids = selected_ids[0: imgs.shape[0] / 10]
    metric_data_eval_std = imgs[selected_ids]

    selected_ids = np.random.permutation(range(imgs.shape[0]))
    selected_ids = selected_ids[0: imgs.shape[0] / 10]
    random_imgs = imgs[selected_ids]
    random_latents = latent_values[selected_ids]

    def discretize(data, num_bins=20):
        """ Adapted from:
            https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/utils.py
        """
        discretized = np.zeros_like(data, dtype=np.int32)
        for i in range(data.shape[1]):
            discretized[:, i] = np.digitize(
                data[:, i],
                np.histogram(data[:, i], num_bins)[1][:-1])
            assert np.min(discretized[:, i]) == 1
            assert np.max(discretized[:, i]) == num_bins
            discretized[:, i] -= 1
        return discretized

    random_latent_ids = discretize(random_latents)

    metric_data_img_with_latent = {
        "img": random_imgs,
        "latent": random_latents,
        "latent_id": random_latent_ids,
        "is_continuous": [True, True, True, True, True]}


    metric_data = {
        "groups": metric_data_groups,
        "img_eval_std": metric_data_eval_std,
        "img_with_latent": metric_data_img_with_latent}

    return imgs, metric_data, latent_values


if __name__ == "__main__":
    load_3Dpots("../data/3Dpots")
