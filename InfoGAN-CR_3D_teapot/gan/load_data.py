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

    metric_data = {
        "groups": metric_data_groups,
        "img_eval_std": metric_data_eval_std}

    return imgs, metric_data, latent_values


if __name__ == "__main__":
    load_3Dpots("../data/3Dpots")
