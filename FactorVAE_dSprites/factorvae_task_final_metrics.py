from gpu_task_scheduler.gpu_task import GPUTask


class FactorVAETask(GPUTask):
    def main(self):
        import os
        import tensorflow as tf
        from gan.load_data import load_dSprites
        from gan.latent import GaussianLatent, JointLatent
        from gan.network import VAEDecoder, VAEEncoder, TCDiscriminator, \
            MetricRegresser
        from gan.factorVAE import FactorVAE
        from gan.metric import FactorVAEMetric, DSpritesInceptionScore, \
            DHSICMetric, \
            BetaVAEMetric, SAPMetric, FStatMetric, MIGMetric, DCIMetric
        import pickle

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

            results = {}

            factorVAEMetric_f = FactorVAEMetric(metric_data, sess=sess)
            factorVAEMetric_f.set_model(vae)
            results["FactorVAE"] = factorVAEMetric_f.evaluate(-1, -1, -1)

            betaVAEMetric_f = BetaVAEMetric(metric_data, sess=sess)
            betaVAEMetric_f.set_model(vae)
            results["betaVAE"] = betaVAEMetric_f.evaluate(-1, -1, -1)
            
            sapMetric_f = SAPMetric(metric_data, sess=sess)
            sapMetric_f.set_model(vae)
            results["SAP"] = sapMetric_f.evaluate(-1, -1, -1)

            fStatMetric_f = FStatMetric(metric_data, sess=sess)
            fStatMetric_f.set_model(vae)
            results["FStat"] = fStatMetric_f.evaluate(-1, -1, -1)

            migMetric_f = MIGMetric(metric_data, sess=sess)
            migMetric_f.set_model(vae)
            results["MIG"] = migMetric_f.evaluate(-1, -1, -1)

            for regressor in ["Lasso", "LassoCV", "RandomForest", "RandomForestIBGAN", "RandomForestCV"]:
                dciVAEMetric_f = DCIMetric(metric_data, sess=sess, regressor=regressor)
                dciVAEMetric_f.set_model(vae)
                results["DCI_{}".format(regressor)] = dciVAEMetric_f.evaluate(-1, -1, -1)

            with open(os.path.join(self._work_dir, "final_metrics.pkl"), "wb") as f:
                pickle.dump(results, f)


