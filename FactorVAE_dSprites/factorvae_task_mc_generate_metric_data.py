from gpu_task_scheduler.gpu_task import GPUTask


class FactorVAETask(GPUTask):
    def main(self):
        import os
        import tensorflow as tf
        import pickle
        from gan.load_data import load_dSprites
        from gan.latent import GaussianLatent, JointLatent
        from gan.network import VAEDecoder, VAEEncoder, TCDiscriminator, \
            MetricRegresser
        from gan.factorVAE import FactorVAE
        from gan.metric import FactorVAEMetric, DSpritesInceptionScore, \
            DHSICMetric

        data, metric_data, latent_values, metadata = \
            load_dSprites("data/dSprites")
        _, height, width, depth = data.shape

        latent_list = []

        for i in range(self._config["gaussian_dim"]):
            latent_list.append(GaussianLatent(
                in_dim=1, out_dim=1, loc=0.0, scale=1.0, q_std=1.0,
                apply_reg=True))
        latent = JointLatent(latent_list=latent_list)

        decoder = VAEDecoder(
            output_width=width, output_height=height, output_depth=depth)
        encoder = VAEEncoder(output_length=latent.reg_in_dim)
        tcDiscriminator = TCDiscriminator()

        shape_network = MetricRegresser(
            output_length=3,
            scope_name="dSpritesSampleQualityMetric_shape")

        checkpoint_dir = os.path.join(self._work_dir, "checkpoint")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        sample_dir = os.path.join(self._work_dir, "sample")
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        time_path = os.path.join(self._work_dir, "time.txt")
        metric_path = os.path.join(self._work_dir, "metric.csv")

        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        with tf.Session(config=run_config) as sess:
            factorVAEMetric = FactorVAEMetric(metric_data, sess=sess)
            dSpritesInceptionScore = DSpritesInceptionScore(
                sess=sess,
                do_training=False,
                data=data,
                metadata=metadata,
                latent_values=latent_values,
                network_path="metric_model/DSprites",
                shape_network=shape_network,
                sample_dir=sample_dir)
            dHSICMetric = DHSICMetric(
                sess=sess,
                data=data)
            metric_callbacks = [factorVAEMetric,
                                dSpritesInceptionScore,
                                dHSICMetric]
            vae = FactorVAE(
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
                encoder=encoder,
                tcDiscriminator=tcDiscriminator,
                tc_coe=self._config["tc_coe"],
                metric_callbacks=metric_callbacks,
                metric_freq=self._config["metric_freq"],
                metric_path=metric_path,
                output_reverse=self._config["output_reverse"])
            vae.build()
            vae.load()

            metric_data_groups = []
            L = 100
            M = 1000

            for i in range(M):
                fixed_latent_id = i % 10
                latents_sampled = vae.latent.sample(L)
                latents_sampled[:, fixed_latent_id] = \
                    latents_sampled[0, fixed_latent_id]
                imgs_sampled = vae.sample_from(latents_sampled)
                metric_data_groups.append(
                    {"img": imgs_sampled,
                     "label": fixed_latent_id})

            latents_sampled = vae.latent.sample(data.shape[0] / 10)
            metric_data_eval_std = vae.sample_from(latents_sampled)

            metric_data = {
                "groups": metric_data_groups,
                "img_eval_std": metric_data_eval_std}

            metric_data_path = os.path.join(self._work_dir, "metric_data.pkl")
            with open(metric_data_path, "wb") as f:
                pickle.dump(metric_data, f, protocol=2)
