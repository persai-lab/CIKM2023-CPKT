from llpkt.utils.config import *
from llpkt.utils.misc import *
from llpkt.agents import *
import numpy as np
from sklearn import metrics
import torch
import time
# from ray import tune
import os
# from functools import partial


def single_exp(json_config, model, data, mode, exp_name_format, args_list, config_dir,
               root_dir=None, **kwargs):
    # print(json_config)
    if root_dir is not None:
        os.chdir(root_dir)
    # print(os.getcwd())
    key = []
    for arg in args_list:
        key.append(json_config[arg])
    key = tuple(key)
    exp_name = exp_name_format.format(*key)

    if "exp_name" in kwargs:
        exp_name += '_' + kwargs["exp_name"]
    if "remark" in kwargs:
        remark = kwargs["remark"]

    json_config["exp_name"] = exp_name
    json_config["mode"] = mode
    # for key in json_config:
    #     print("{}: {}".format(key, json_config[key]))
    config_file_path = "{}/{}.json".format(config_dir, exp_name)
    json.dump(json_config, open(config_file_path, "w"))
    config = process_config(config_file_path)

    # print("all globals *************************************************{}".format(globals()))
    agent_class = globals()[config.agent]
    agent = agent_class(config)
    agent.run()

    result_dir_path = "experiments/{}Agent".format(model)
    create_dirs([result_dir_path])
    # print("Result file dir created at: {}".format(result_dir_path))
    if mode in ["hyperparameters"]:
        size, perf = agent.finalize()
        config["size"] = size
        config["perf"] = perf
        # tune.report(perf=perf, size=size)

        result_file_path = "{}/{}_{}{}.json".format(result_dir_path, mode, data, remark)
        if not os.path.exists(result_file_path):
            with open(result_file_path, "w") as f:
                pass
        with open(result_file_path, "a") as f:
            f.write(json.dumps(config) + "\n")
    elif mode == "online_test":
        result_file_path = "{}/{}_{}{}.csv".format(result_dir_path, mode, data, remark)
        if not os.path.exists(result_file_path):
            with open(result_file_path, "w") as f:
                curr_time = time.time()
                f.write("Time: {}\n".format(curr_time))

        true_labels = torch.load(config.out_dir + "{}_true_labels.tar".format(mode))
        pred_labels = torch.load(config.out_dir + "{}_pred_labels.tar".format(mode))
        size = len(true_labels)
        if config.metric == "rmse":
            rmse = np.sqrt(metrics.mean_squared_error(true_labels, pred_labels))
            mae = metrics.mean_absolute_error(true_labels, pred_labels)
            with open(result_file_path, "a") as f:
                f.write("{},{},{},{}\n".format(exp_name, size, rmse, mae))
            print("{},Size: {}, RMSE: {}, MAE: {}\n".format(exp_name, len(true_labels), rmse, mae))
        elif config.metric == "auc":
            roc_auc = metrics.roc_auc_score(true_labels, pred_labels)
            prec, rec, _ = metrics.precision_recall_curve(true_labels, pred_labels)
            pr_auc = metrics.auc(rec, prec)
            with open(result_file_path, "a") as f:
                f.write("{},{},{},{}\n".format(exp_name, size, roc_auc, pr_auc))
        else:
            raise AttributeError
    else:
        raise ValueError


def check_progress(model, data, args_list, remark=""):
    metric = None
    result_file_dir = "experiments/{}Agent".format(model)
    result_file_path = "{}/hyperparameters_{}{}.json".format(result_file_dir, data, remark)

    progress_dict = {}
    best_config = {}
    performance_dict = {}
    para_config_mapping = {}
    duplicate = 0
    if not os.path.exists(result_file_path):
        create_dirs([result_file_dir])
        with open(result_file_path, "w") as f:
            pass
        return progress_dict, best_config

    with open(result_file_path, "r") as f:
        for line in f:
            try:
                result = json.loads(line)
                metric = result['metric']
            except json.decoder.JSONDecodeError:
                print(line)

            key = []
            for arg in args_list:
                key.append(result[arg])
            key = tuple(key)
            if key not in progress_dict:
                progress_dict[key] = True
            else:
                duplicate += 1
                pass
                # print("duplicate: {}".format(key))
                # print(line)
            if key not in performance_dict:
                # performance_dict[key] = (
                #     result["best_epoch"], result["best_train_loss"], result["best_val_perf"]
                # )
                performance_dict[key] = (result["perf"], result["size"])
            if key not in para_config_mapping:
                para_config_mapping[key] = result


    if metric is None:
        return progress_dict, best_config
    elif metric == "auc":
        sorted_perf = sorted(performance_dict.items(), key=lambda x: x[1][0], reverse=True)
        best_key = sorted_perf[0][0]
        best_config = para_config_mapping[best_key]
    elif metric == "rmse":
        sorted_perf = sorted(performance_dict.items(), key=lambda x: x[1][0])
        best_key = sorted_perf[0][0]
        best_config = para_config_mapping[best_key]
    else:
        raise AttributeError

    for arg in args_list:
        print("{},".format(arg), end="")
    print("size, perf ({})".format(metric))
    for (para, (perf, size)) in sorted_perf[::-1]:
        for k in para:
            print("{},".format(k), end="")
        print("{},{}".format(size, perf))
    for arg in args_list:
        print("{},".format(arg), end="")
    print("size, perf ({})".format(metric))
    print("\nNumber of existing records: {}, duplicate: {}".format(len(progress_dict), duplicate))
    #
    # for i, arg in enumerate(args_list):
    #     best_config[arg] = sorted_perf[0][0][i]
    # if "stride" in result:
    #     best_config.pop("stride")
    # # best_config["target_train_loss"] = best_train_loss  # used to early stop on training
    # # best_config.pop("best_epoch")
    # # best_config.pop("best_train_loss")
    # # best_config.pop("best_val_perf")
    # best_config.pop("summary_dir")
    # best_config.pop("checkpoint_dir")
    # best_config.pop("out_dir")
    # best_config.pop("log_dir")
    # print(best_config)
    return progress_dict, best_config


def test_5folds(fold, best_config, model, data, mode, exp_name_format, args_list, root_dir=None,
                remark=""):
    config_dir = "configs/5folds/{}/{}".format(model, data)
    create_dirs([config_dir])
    best_config["mode"] = mode
    prGreen("\nfold {}".format(fold))
    exp_name = "{}_{}_fold_{}".format(model, data, fold)
    best_config["data_name"] = "{}_fold_{}".format(data, fold)
    print(best_config)
    single_exp(best_config, model, data, mode, exp_name_format, args_list, config_dir, root_dir,
               exp_name=exp_name, remark=remark)

    # analysis = tune.run(
    #     partial(single_exp, model=model, data=data, mode=mode, exp_name_format=exp_name_format,
    #             args_list=args_list, config_dir=config_dir, root_dir=root_dir, remark=remark),
    #             resources_per_trial={"cpu": 1, "gpu": 0},
    #             config=best_config
    # )
