import os
from metric import DSpritesInceptionScore
from load_data import load_dSprites
from network import MetricRegresser
import tensorflow as tf

if __name__ == "__main__":
    data, metric_data, latent_values, metadata = \
        load_dSprites("../data/dSprites")

    network_path = "../metric_model/DSprites"
    if not os.path.exists(network_path):
        os.makedirs(network_path)

    checkpoint_dir = "../metric_model/DSprites/checkpoint"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    shape_network = MetricRegresser(
        output_length=3,
        scope_name="dSpritesSampleQualityMetric_shape")

    run_config = tf.ConfigProto()

    with tf.Session(config=run_config) as sess:
        dSpritesSampleQualityMetric = DSpritesInceptionScore(
            sess=sess,
            do_training=True,
            data=data,
            metadata=metadata,
            latent_values=latent_values,
            network_path=network_path,
            checkpoint_dir=checkpoint_dir,
            shape_network=shape_network)
        dSpritesSampleQualityMetric.build()
        dSpritesSampleQualityMetric.train()
