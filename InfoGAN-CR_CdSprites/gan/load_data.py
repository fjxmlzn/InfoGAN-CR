import os
import numpy as np


def load_CdSprites(path):
    # part of the code is from:
    # https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_reloading_example.ipynb
    dataset_zip = np.load(
        os.path.join(
            path, "cdsprites_ndarray_64x64.npz"),
        allow_pickle=True)
    imgs = dataset_zip['imgs']
    latent_values = dataset_zip['latents_values']
    #latents_classes = dataset_zip['latents_classes']
    metadata = dataset_zip['metadata'][()]

    imgs = imgs.reshape(-1, 64, 64, 1).astype(np.float)  # 0 ~ 1

    latents_names = metadata["latents_names"]
    latents_sizes = metadata["latents_sizes"]
    latents_possible_values = metadata["latents_possible_values"]
    latents_bases = np.concatenate(
        (latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))

    def latent_to_index(latents):
        return np.dot(latents, latents_bases).astype(int)

    def sample_latent(size=1):
        samples = np.zeros((size, latents_sizes.size))
        for lat_i, lat_size in enumerate(latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)
        return samples

    metric_data_groups = []
    L = 25
    M = 200

    for i in range(M):
        fixed_latent_id = i % latents_sizes.shape[0]
        latents_sampled = sample_latent(size=L)
        latents_sampled[:, fixed_latent_id] = \
            np.random.randint(latents_sizes[fixed_latent_id], size=1)
        # print(latents_sampled[0:10])
        indices_sampled = latent_to_index(latents_sampled)
        imgs_sampled = imgs[indices_sampled]
        metric_data_groups.append(
            {"img": imgs_sampled,
             "label": fixed_latent_id})

    selected_ids = np.random.permutation(range(imgs.shape[0]))
    selected_ids = selected_ids[0: imgs.shape[0] / 10]
    metric_data_eval_std = imgs[selected_ids]

    random_latent_ids = sample_latent(size=imgs.shape[0] / 10)
    random_latent_ids = random_latent_ids.astype(np.int32)
    random_ids = latent_to_index(random_latent_ids)
    assert random_latent_ids.shape == (imgs.shape[0] / 10, latents_sizes.shape[0])
    random_imgs = imgs[random_ids]

    random_latents = np.zeros((random_imgs.shape[0], latents_sizes.shape[0]))
    for i in range(latents_sizes.shape[0]):
        random_latents[:, i] = \
            latents_possible_values[latents_names[i]][random_latent_ids[:, i]]

    metric_data_img_with_latent = {
        "img": random_imgs,
        "latent": random_latents,
        "latent_id": random_latent_ids,
        "is_continuous": [True,]*latents_sizes.shape[0]}

    metric_data = {
        "groups": metric_data_groups,
        "img_eval_std": metric_data_eval_std,
        "img_with_latent": metric_data_img_with_latent}

    return imgs, metric_data, latent_values, metadata


if __name__ == "__main__":
    load_dSprites("../data/CdSprites")
