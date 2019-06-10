from gpu_task_scheduler.gpu_task import GPUTask


class GANTask(GPUTask):
    def main(self):
        import os
        import tensorflow as tf
        from gan.load_data import load_dSprites
        from gan.latent import UniformLatent, JointLatent
        from gan.network import Decoder, InfoGANDiscriminator, \
            CrDiscriminator, MetricRegresser
        from gan.infogan_cr import INFOGAN_CR
        from gan.metric import FactorVAEMetric, DSpritesInceptionScore, \
            DHSICMetric

        data, metric_data, latent_values, metadata = \
            load_dSprites("data/dSprites")
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
            InfoGANDiscriminator(
                output_length=latent.reg_out_dim,
                q_l_dim=self._config["q_l_dim"])
        crDiscriminator = CrDiscriminator(output_length=latent.num_reg_latent)

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
            gan.train()
