from exps.exp_config import *
import argparse
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler
from functools import partial
from pathlib import Path
from llpkt.utils.misc import *
from llpkt.utils.dirs import *


# from ray.tune.suggest.bayesopt import BayesOptSearch
# import nevergrad as ng
# from ray.tune.suggest.nevergrad import NevergradSearch


def hyperparameters_tuning(model, data, args_list, exp_name_format, root_dir=None, num_samples=10,
                           cpu_per_trial=1, remark=""):
    max_offline_epochs = 30
    max_epochs = 10

    test_size = 50
    hist_size = 50
    peek_size = 0
    hidden_dim = 128
    max_seq_len = test_size + hist_size + peek_size
    if data == "MORF686":
        metric = "rmse"
        test_start_index = tune.grid_search([5])
        min_concepts = 2
        max_concepts = 20
        num_concepts = tune.grid_search([9])
        # user_embed_dim = tune.grid_search([1, 16, 2, 8, 4])
        user_embed_dim = hidden_dim
        val_freq = tune.grid_search([5, 7])
    else:
        metric = "auc"
        test_start_index = 50
        min_concepts = 2
        max_concepts = 50
        num_concepts = tune.sample_from(lambda _: np.random.randint(min_concepts, max_concepts))
        # num_concepts = tune.grid_search(list(range(2, 100)))
        # num_concepts = 6
        # hidden_dim = tune.grid_search([16, 32, 64, 128])
        # user_embed_dim = tune.grid_search([1, 2, 4])
        user_embed_dim = 2
        val_freq = tune.grid_search([10, 30])

    mode = "hyperparameters"
    config_dir = os.path.abspath("configs/{}/{}".format(model, data))
    create_dirs([config_dir])
    config = {
        "agent": "{}Agent".format(model),
        "metric": metric,
        "cuda": False,
        "seed": 123,
        "data_name": data,
        "min_seq_len": 2,
        "test_size": test_size,
        "hist_size": hist_size,
        "peek_size": peek_size,
        "val_freq": val_freq,
        "max_seq_len": max_seq_len,
        "test_start_index": test_start_index,
        "validation_split": 0.2,
        "shuffle": True,
        "num_workers": 1,
        "dropout_rate": tune.grid_search([0., 0.1]),
        # "dropout_rate": 0.,
        "batch_size": tune.grid_search([32, 64, 128, 256, 512]),
        # "batch_size": 64,
        "num_concepts": num_concepts,
        "user_embed_dim": user_embed_dim,
        "hidden_dim": hidden_dim,
        "sse_user_prob": tune.grid_search([0.1, 0.3, 0.5]),
        # "sse_user_prob": 0.3,
        "sse_item_prob": tune.grid_search([0.1, 0.3, 0.5]),
        # "sse_item_prob": 0.1,
        "sse_type": 'ltm',
        "optimizer": "adam",
        "momentum": 0.9,
        "weight_decay": tune.grid_search([0., 0.1, 0.01, 0.001]),
        # "weight_decay": 0.,
        "learning_rate": 0.01,
        "epsilon": 0.1,
        "max_grad_norm": 10.,
        "max_epoch": max_epochs,
        "max_offline_epoch": max_offline_epochs,
        "log_interval": 10,
        "validate_every": 1,
        "offline_testing": False,
        "save_checkpoint": False,
        "checkpoint_file": "checkpoint.pth.tar"
    }

    # ng_search = NevergradSearch(optimizer=ng.optimizers.OnePlusOne,
    #                             metric="perf",
    #                             mode="min")
    reporter = CLIReporter(parameter_columns=args_list,
                           metric_columns=["offline_t_perf", "offline_t_size", "t_perf",
                                           "t_size", "t_index"],
                           max_progress_rows=100,
                           sort_by_metric=True,
                           max_report_frequency=10)
    reporter.add_metric_column("perf")
    reporter.add_metric_column("size")
    reporter.add_metric_column("epoch")
    analysis = tune.run(
        partial(single_exp, model=model, data=data, mode=mode, exp_name_format=exp_name_format,
                args_list=args_list, config_dir=config_dir, root_dir=root_dir, remark=remark),
        resources_per_trial={"cpu": cpu_per_trial, "gpu": 0},
        config=config,
        num_samples=num_samples,
        local_dir="./ray_tune_results",
        progress_reporter=reporter
        # search_alg=ng_search
    )


if __name__ == '__main__':
    working_dir = os.path.abspath(".")

    personalized = True
    # personalized = False

    if personalized:
        model = "LLPKTFast"
    else:
        model = "LLKTFast"
    args_list = ["seed", "batch_size", "test_start_index", "test_size", "hist_size", "peek_size",
                 "user_embed_dim", "hidden_dim", "num_concepts", "weight_decay", "sse_user_prob",
                 "sse_item_prob", "val_freq", "dropout_rate", "learning_rate", "max_grad_norm"]
    exp_name_format = "exp_seed_{}_bs_{}_tsi_{}_ts_{}_hs_{}_ps_{}_ued_{}_hd_{}_nc_{}_wd_{}_" \
                      "sup_{}_sip_{}_vf_{}_dp_{}_lr_{}_mgn_{}"
    # args_list = ["seed", "batch_size", "test_start_index", "test_size", "hist_size",
    #              "user_embed_dim", "hidden_dim", "num_concepts", "weight_decay", "sse_user_prob",
    #              "sse_item_prob", "val_freq", "dropout_rate", "learning_rate", "max_grad_norm"]
    # exp_name_format = "exp_seed_{}_bs_{}_tsi_{}_ts_{}_hs_{}_ued_{}_hd_{}_nc_{}_wd_{}_" \
    #                   "sup_{}_sip_{}_vf_{}_dp_{}_lr_{}_mgn_{}"

    # data = "MORF686"
    # data = "ASSISTments2015"
    data = "EdNetLec"
    # data = "Junyi1564"

    # exp_mode = "hyperparameters"
    # exp_mode = "check_progress"
    exp_mode = "online_test"

    # ATTENTION: ****************** REMEMBER to modify the model accordingly!!!!!!!!!!!!!! ********
    # remark = "_{}_offline".format(model)
    remark = "_{}_offline_train_val_test".format(model)
    # remark = "_{}".format(model)
    prPurple("MODE: {}, Model: {}, Data: {} Remark: {}".format(exp_mode, model, data, remark))

    arg_parser = argparse.ArgumentParser(description="LLPKT Experiments")
    # arg_parser.add_argument('-sl', '--seq_len', type=int, default=50)
    # arg_parser.add_argument('-bs', '--batch_size', type=int, default=16)
    # arg_parser.add_argument('-hd', '--hidden_dim', type=int, default=32)
    args = arg_parser.parse_args()

    if exp_mode == "hyperparameters":
        # num_samples = 1 means 1 round of grid-search parameters
        hyperparameters_tuning(model, data, args_list, exp_name_format, working_dir,
                               num_samples=1, cpu_per_trial=1, remark=remark)

    elif exp_mode == "online_test":
        if data == "MORF686":
            seed = 1024
            metric = "rmse"
            batch_size = 32
            test_start_index = 5
            test_size = 15
            hist_size = 5
            peek_size = 0
            # it is best to set test_size == hist_size
            val_freq = 15
            max_seq_len = test_size + hist_size + peek_size
            user_embed_dim = 1
            hidden_dim = 32
            num_concepts = 11
            weight_decay = 0.
            sse_user_prob = 0.
            sse_item_prob = 0.1
            # sse_user_prob = 0.
            # sse_item_prob = 0.
            dropout_rate = 0.
            sse_type = 'ltm'
            learning_rate = 0.01
            max_grad_norm = 10.
            optimizer = "adam"
        elif data == "ASSISTments2015":
            seed = 123
            metric = "auc"
            batch_size = 512
            test_start_index = 50
            test_size = 10
            hist_size = 100
            peek_size = 40
            # it is best to set test_size == hist_size
            val_freq = 10
            max_seq_len = test_size + hist_size + peek_size
            user_embed_dim = 2
            hidden_dim = 128
            num_concepts = 6
            weight_decay = 0.
            # sse_user_prob = 0.5
            # sse_item_prob = 0.1
            sse_user_prob = 0.
            sse_item_prob = 0.
            dropout_rate = 0.
            sse_type = 'ltm'
            learning_rate = 0.01
            max_grad_norm = 10.
            optimizer = "adam"
        elif data == "Junyi1564":
            seed = 123
            # seed = 959
            metric = "auc"
            batch_size = 32
            test_start_index = 100
            test_size = 10
            hist_size = 150
            peek_size = 40
            val_freq = 10
            max_seq_len = test_size + hist_size + peek_size
            user_embed_dim = 2
            hidden_dim = 128
            num_concepts = 7
            weight_decay = 0.
            sse_user_prob = 0.3
            sse_item_prob = 0.1
            # sse_user_prob = 0.
            # sse_item_prob = 0.
            dropout_rate = 0.
            sse_type = 'ltm'
            learning_rate = 0.01
            max_grad_norm = 10.
            optimizer = "adam"
        elif data == "EdNetLec":
            seed = 123
            metric = "auc"
            batch_size = 32
            test_start_index = 100
            test_size = 10
            hist_size = 40
            peek_size = 0
            val_freq = 10
            max_seq_len = test_size + hist_size + peek_size
            user_embed_dim = 4
            hidden_dim = 128
            num_concepts = 41
            weight_decay = 0.1
            sse_user_prob = 0.3
            sse_item_prob = 0.1
            # sse_user_prob = 0.
            # sse_item_prob = 0.
            dropout_rate = 0.1
            sse_type = 'ltm'
            learning_rate = 0.01
            max_grad_norm = 10.
            optimizer = "adam"
            max_epoch = 10
            max_offline_epoch = 30
        else:
            raise ValueError

        best_config = {
            "agent": "{}Agent".format(model),
            "metric": metric,
            "cuda": False,
            "seed": seed,
            "data_name": data,
            "min_seq_len": 2,
            "max_seq_len": max_seq_len,
            "test_start_index": test_start_index,
            "test_size": test_size,
            "hist_size": hist_size,
            "peek_size": peek_size,
            "val_freq": val_freq,
            "validation_split": 0.2,
            "shuffle": True,
            "num_workers": 8,
            "dropout_rate": dropout_rate,
            "batch_size": batch_size,
            "user_embed_dim": user_embed_dim,
            "hidden_dim": hidden_dim,
            "num_concepts": num_concepts,
            "optimizer": optimizer,
            "momentum": 0.9,
            "weight_decay": weight_decay,
            "sse_user_prob": sse_user_prob,
            "sse_item_prob": sse_item_prob,
            "sse_type": sse_type,
            "learning_rate": learning_rate,
            "epsilon": 0.1,
            "max_grad_norm": max_grad_norm,
            "max_epoch": 10,
            "max_offline_epoch": 30,
            "log_interval": 10,
            "validate_every": 1,
            "offline_testing": False,
            "save_checkpoint": True,
            "checkpoint_file": "checkpoint.pth.tar"
        }

        mode = "online_test"
        fold = 5
        test_5folds(fold, best_config, model, data, mode, exp_name_format, args_list,
                    root_dir=working_dir, remark=remark)
    else:
        progress_dict, best_config = check_progress(model, data, args_list, remark=remark)
