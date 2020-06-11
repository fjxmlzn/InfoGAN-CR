config = {
    "scheduler_config": {
        "gpu": ["0", "1", "2", "3"],
        "temp_folder": "temp",
        "scheduler_log_file_path": "scheduler.log",
        "log_file": "worker.log"
    },

    "global_config": {
        "epoch": 27,
        "batch_size": 64,
        "vis_freq": 200,
        "vis_num_sample": 10,
        "vis_num_rep": 10,
        "metric_freq": 400,
        "output_reverse": False
    },

    "test_config": [
        {
            "run": range(10),
            "gaussian_dim": [10],
            "tc_coe": [1.0, 10.0, 20.0, 40.0]
        }
    ]
}
