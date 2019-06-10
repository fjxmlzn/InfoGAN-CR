config = {
    "scheduler_config": {
        "gpu": ["0", "1", "2", "3"],
        "temp_folder": "temp",
        "scheduler_log_file_path": "scheduler.log",
        "log_file": "worker.log"
    },

    "global_config": {
        "epoch": 28,
        "batch_size": 64,
        "vis_freq": 200,
        "vis_num_sample": 10,
        "vis_num_rep": 10,
        "metric_freq": 400,
        "output_reverse": False,
        "uniform_reg_dim": 5,
        "uniform_not_reg_dim": 5,
        "de_lr": 0.001,
        "infod_lr": 0.002,
        "crd_lr": 0.002,
        "q_l_dim": 128,
        "summary_freq": 1
    },

    "test_config": [
        {
            "run": range(10),
            "info_coe_de": [0.05],
            "info_coe_infod": [0.05],
            "gap_start": [0.0],
            "gap_decrease_times": [0],
            "gap_decrease": [0.0],
            "gap_decrease_batch": [1],
            "cr_coe_start": [0.0],
            "cr_coe_increase_times": [1],
            "cr_coe_increase": [2.0],
            "cr_coe_increase_batch": [288000]
        }
    ]
}
