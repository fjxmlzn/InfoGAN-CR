config = {
    "scheduler_config": {
        "gpu": ["0"],
        "temp_folder": "temp",
        "scheduler_log_file_path": "scheduler_generate_latent_trans.log",
        "log_file": "worker_generate_latent_trans.log",
        "force_rerun": True
    },

    "global_config": {
        "epoch": 57,
        "batch_size": 128,
        "vis_freq": 200,
        "vis_num_sample": 10,
        "vis_num_rep": 10,
        "metric_freq": 400,
        "output_reverse": False,
        "uniform_reg_dim": 5,
        "uniform_not_reg_dim": 100,
        "de_lr": 2e-3,
        "infod_lr": 2e-4,
        "crd_lr": 2e-4,
        "summary_freq": 1,
    },

    "test_config": [
        {
            "run": [0],
            "info_coe_de": [2.0],
            "info_coe_infod": [2.0],
            "gap_start": [0.0],
            "gap_decrease_times": [0],
            "gap_decrease": [0.0],
            "gap_decrease_batch": [1],
            "cr_coe_start": [0.0],
            "cr_coe_increase_times": [1],
            "cr_coe_increase": [1.0],
            "cr_coe_increase_batch": [80000]
        }
    ]
}
