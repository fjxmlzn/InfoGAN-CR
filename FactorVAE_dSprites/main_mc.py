import csv
import numpy as np
import os
import pickle
from config_mc import config
from gpu_task_scheduler.config_manager import ConfigManager

METRICS = [
    ("FactorVAE", ["factorVAE_metric"]),
    ("betaVAE", ["betaVAE_metric"]),
    ("SAP", ["SAP_metric"]),
    ("FStat", ["FStat_modu_metric", "FStat_expl_metric"]),
    ("MIG", ["MIG_metric"]),
    ("DCI_Lasso", ["DCI_Lasso_disent_metric", "DCI_Lasso_complete_metric"]),
    ("DCI_LassoCV", ["DCI_LassoCV_disent_metric",
                     "DCI_LassoCV_complete_metric"]),
    ("DCI_RandomForest", ["DCI_RandomForest_disent_metric",
                          "DCI_RandomForest_complete_metric"]),
    ("DCI_RandomForestIBGAN", ["DCI_RandomForestIBGAN_disent_metric",
                               "DCI_RandomForestIBGAN_complete_metric"]),
    ("DCI_RandomForestCV", ["DCI_RandomForestCV_disent_metric",
                            "DCI_RandomForestCV_complete_metric"])]


def get_metrics():
    ori_metrics = []
    ori_tc = []
    ori_other_metrics = []
    config_manager = ConfigManager(config)
    num_runs = config_manager.get_num_left_config()
    while config_manager.get_num_left_config() > 0:
        cur_config = config_manager.get_next_config()
        _config = cur_config["config"]
        ori_tc.append(_config["tc_coe"])
        _work_dir = cur_config["work_dir"]
        metric_path = os.path.join(_work_dir, "metric.csv")
        with open(metric_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["epoch_id"] == "26" and row["batch_id"] == "-1":
                    ori_metrics.append(float(row["factorVAE_metric"]))
        with open(os.path.join(_work_dir, "final_metrics.pkl"), "rb") as f:
            other_metrics = pickle.load(f)
        ori_other_metrics.append(other_metrics)
    assert len(ori_metrics) == num_runs

    return ori_metrics, ori_tc, ori_other_metrics


def get_cross_metrics():
    metric_values = []
    metric_names = []

    # get cross results
    config_manager = ConfigManager(config)
    num_runs = config_manager.get_num_left_config()
    cross_metrics = np.zeros((num_runs, num_runs))
    all_configs = []
    i = 0
    while config_manager.get_num_left_config() > 0:
        cur_config = config_manager.get_next_config()
        _work_dir = cur_config["work_dir"]
        file_path = os.path.join(_work_dir, "cross_evaluation.pkl")
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        results = data["results"]
        configs = data["configs"]
        all_configs.append(configs)

        for j in range(num_runs):
            cross_metrics[i, j] = results[j]["factorVAE_metric"]

        i += 1

    # double check the configs
    for i in range(len(all_configs)):
        for j in range(len(all_configs[0])):
            for k in all_configs[0][0]:
                assert all_configs[i][j][k] == all_configs[0][j][k]

    cross_metrics_average = (cross_metrics + cross_metrics.T) / 2

    return metric_values, metric_names, cross_metrics, cross_metrics_average


def print_sort_model():
    config_manager = ConfigManager(config)
    num_runs = config_manager.get_num_left_config()

    # get original metric
    ori_metrics, ori_tc, ori_other_metrics = get_metrics()

    # get cross results
    metric_values, metric_names, cross_metrics, cross_metrics_average = \
        get_cross_metrics()

    sort_list = []
    for i in range(num_runs):
        array = (list(cross_metrics_average[i, :i]) +
                 list(cross_metrics_average[i, i + 1:]))
        sort_list.append(np.mean(array))

    sort_order = np.argsort(sort_list)[::-1]

    for i in range(num_runs):
        other_metric_str = ""
        for metric in METRICS:
            for sub_metric in metric[1]:
                other_metric_str += "{}={}, ".format(
                    sub_metric,
                    ori_other_metrics[sort_order[i]][metric[0]][sub_metric])
        print("{:0.3f} ({}, tc={}, {})".format(
            sort_list[sort_order[i]],
            ori_metrics[sort_order[i]],
            ori_tc[sort_order[i]],
            other_metric_str))


if __name__ == "__main__":
    print_sort_model()
