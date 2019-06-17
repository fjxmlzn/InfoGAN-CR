from gpu_task_scheduler.gpu_task import GPUTask


class GANTask(GPUTask):
    def main(self):
        import os
        import tensorflow as tf
        from gan.load_data import load_celeba
        from gan.latent import UniformLatent, JointLatent
        from gan.network import Decoder, InfoGANDiscriminator, \
            CrDiscriminator
        from gan.infogan_cr import INFOGAN_CR
        import numpy as np
        import imageio

        data = load_celeba("data/celeba")
        _, height, width, depth = data.shape

        latent_list = []

        for i in range(self._config["uniform_reg_dim"]):
            latent_list.append(UniformLatent(
                in_dim=1, out_dim=1, low=-1.0, high=1.0, q_std=1.0,
                apply_reg=True))
        if self._config["uniform_not_reg_dim"] > 0:
            latent_list.append(UniformLatent(
                in_dim=self._config["uniform_not_reg_dim"],
                out_dim=self._config["uniform_not_reg_dim"],
                low=-1.0, high=1.0, q_std=1.0,
                apply_reg=False))
        latent = JointLatent(latent_list=latent_list)

        decoder = Decoder(
            output_width=width, output_height=height, output_depth=depth)
        infoGANDiscriminator = \
            InfoGANDiscriminator(output_length=latent.reg_out_dim)
        crDiscriminator = \
            CrDiscriminator(output_length=latent.num_reg_latent)

        checkpoint_dir = os.path.join(self._work_dir, "checkpoint")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        sample_dir = os.path.join(self._work_dir, "sample")
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        time_path = os.path.join(self._work_dir, "time.txt")
        metric_path = os.path.join(self._work_dir, "metric.csv")

        run_config = tf.ConfigProto()
        with tf.Session(config=run_config) as sess:
            metric_callbacks = []
            gan = INFOGAN_CR(
                sess=sess,
                checkpoint_dir=checkpoint_dir,
                sample_dir=sample_dir,
                time_path=time_path,
                epoch=self._config["epoch"],
                batch_size=self._config["batch_size"],
                data=data,
                vis_freq=self._config["vis_freq"],
                vis_num_sample=self._config["vis_num_sample"],
                vis_num_rep=self._config["vis_num_rep"],
                latent=latent,
                decoder=decoder,
                infoGANDiscriminator=infoGANDiscriminator,
                crDiscriminator=crDiscriminator,
                gap_start=self._config["gap_start"],
                gap_decrease_times=self._config["gap_decrease_times"],
                gap_decrease=self._config["gap_decrease"],
                gap_decrease_batch=self._config["gap_decrease_batch"],
                cr_coe_start=self._config["cr_coe_start"],
                cr_coe_increase_times=self._config["cr_coe_increase_times"],
                cr_coe_increase=self._config["cr_coe_increase"],
                cr_coe_increase_batch=self._config["cr_coe_increase_batch"],
                info_coe_de=self._config["info_coe_de"],
                info_coe_infod=self._config["info_coe_infod"],
                metric_callbacks=metric_callbacks,
                metric_freq=self._config["metric_freq"],
                metric_path=metric_path,
                output_reverse=self._config["output_reverse"],
                de_lr=self._config["de_lr"],
                infod_lr=self._config["infod_lr"],
                crd_lr=self._config["crd_lr"],
                summary_freq=self._config["summary_freq"])
            gan.build()
            gan.load()

            NUM_SAMPLE = 15
            NUM_REP = 100
            GAP = 2

            height, width, depth = gan.image_dims

            latents = gan.latent.uniformly_sample(NUM_SAMPLE, NUM_REP)
            for latent_i, latent in enumerate(latents):
                folder = os.path.join(
                    self._work_dir,
                    "latent_trans",
                    "latent{}".format(latent_i))
                if not os.path.exists(folder):
                    os.makedirs(folder)
                samples = gan.sample_from(latent)

                for i in range(NUM_REP):
                    image = np.zeros((
                        height,
                        NUM_SAMPLE * width + (NUM_SAMPLE - 1) * GAP,
                        depth))
                    for j in range(NUM_SAMPLE):
                        id_ = j * NUM_REP + i
                        col_start = j * (width + GAP)
                        image[:, col_start: col_start + width, :] = \
                            samples[id_]
                    v_min = image.min() - gan.EPS
                    v_max = image.max() + gan.EPS
                    image = (image - v_min) / (v_max - v_min) * 255.0
                    image = image.astype(np.uint8)
                    for j in range(NUM_SAMPLE - 1):
                        col_start = (j + 1) * width + j * GAP
                        image[:, col_start: col_start + GAP, :] = 255
                    file_path = os.path.join(
                        folder,
                        "sample{}.png".format(i))
                    imageio.imwrite(file_path, image)
