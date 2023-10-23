"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
import logging
import torch
import shutil
import numpy as np
from sklearn import metrics
import pickle
import torch.nn as nn

# from tensorboardX.writer import SummaryWriter
from llpkt.utils.metrics import AverageMeter, AverageMeterList


class BaseAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")
        self.current_epoch = None
        self.current_iteration = None
        self.model = None
        self.optimizer = None
        self.data_loader = None

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")
            pass

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = config.seed
        self.mode = config.mode
        self.device = torch.device("cpu")

        # Summary Writer
        self.summary_writer = None
        self.true_labels = None
        self.pred_labels = None
        self.user_outputs = None
        self.best_epoch = None
        self.train_loss = None
        self.train_loss_list = []
        self.best_train_loss = None
        self.best_val_perf = None
        self.metric = config.metric
        self.save = config.save_checkpoint
        if self.metric == "rmse":
            self.best_val_perf = 1.
        elif self.metric == "auc":
            self.best_val_perf = 0.
        else:
            raise AttributeError
        if "target_train_loss" in config:
            self.target_train_loss = config.target_train_loss
        else:
            self.target_train_loss = None

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        filename = self.config.checkpoint_dir + file_name
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            # # self.current_epoch = checkpoint['epoch']
            # # self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            # self.logger.info(f"Checkpoint loaded successfully from '{self.config.checkpoint_dir}'"
            #                  f"at (epoch {checkpoint['epoch']}) at (iteration "
            #                  f"{checkpoint['iteration']})\n")
        except OSError as e:
            self.logger.info(f"No checkpoint exists from '{self.config.checkpoint_dir}'. "
                             f"Skipping...")
            self.logger.info("**First time to train**")
            self.logger.info(f"No checkpoint exists from '{self.config.checkpoint_dir}'. "
                             f"Skipping...")
            self.logger.info("**First time to train**")
            pass

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is
            the best so far
        :return:
        """
        state = {
            'state_dict': self.model.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + file_name)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + file_name,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def save_results(self):
        torch.save(self.true_labels, self.config.out_dir + "{}_true_labels.tar".format(self.mode))
        torch.save(self.pred_labels, self.config.out_dir + "{}_pred_labels.tar".format(self.mode))
        torch.save(self.user_outputs, self.config.out_dir + "{}_user_outputs.tar".format(self.mode))

    def save_pretrain_results(self):
        torch.save(self.true_labels, self.config.out_dir + "pretrain_true_labels.tar")
        torch.save(self.pred_labels, self.config.out_dir + "pretrain_pred_labels.tar")

    def track_best(self, true_labels, pred_labels):
        """
        track the best validation performance and corresponding training error
        :param true_labels:
        :param pred_labels:
        :return:
        """
        self.pred_labels = np.array(pred_labels).squeeze()
        self.true_labels = np.array(true_labels).squeeze()
        self.logger.info("pred size: {} true size {}".format(self.pred_labels.shape, self.true_labels.shape))
        if self.metric == "rmse":
            perf = np.sqrt(metrics.mean_squared_error(self.true_labels, self.pred_labels))
            # self.logger.info('RMSE: {:.05}'.format(perf))
            if perf < self.best_val_perf:
                self.best_val_perf = perf
                self.best_train_loss = self.train_loss.item()
                self.best_epoch = self.current_epoch
        elif self.metric == "auc":
            perf = metrics.roc_auc_score(self.true_labels, self.pred_labels)
            prec, rec, _ = metrics.precision_recall_curve(self.true_labels, self.pred_labels)
            pr_auc = metrics.auc(rec, prec)
            # self.logger.info('ROC-AUC: {:.05}'.format(perf))
            # self.logger.info('PR-AUC: {:.05}'.format(pr_auc))
            if perf > self.best_val_perf:
                self.best_val_perf = perf
                self.best_train_loss = self.train_loss.item()
                self.best_epoch = self.current_epoch
        else:
            raise AttributeError

    def early_stopping(self):
        """
        when we do testing, we would like the training to be early stopped when the training error
        reach the best training loss obtained from validation phase
        :return:
        """
        if self.mode == "test":
            if self.target_train_loss is not None and self.train_loss <= self.target_train_loss:
                # early stop, target train loss comes from hyperparameters tuning step.
                self.logger.info("Early stopping...")
                self.logger.info("Target Train Loss: {}".format(self.target_train_loss))
                self.logger.info("Current Train Loss: {}".format(self.train_loss))
                return True
            # elif self.current_epoch > 10:
            #     if self.train_loss > torch.mean(self.train_loss_list[-10:]):
            #         return True
            # else:
            #     self.train_loss_list.append(self.train_loss)

    def run(self):
        """
        The main operator
        :return:
        """
        if self.mode in ["hyperparameters", "online_test"]:
            try:
                self.train()
            except KeyboardInterrupt:
                # self.logger.info("You have entered CTRL+C.. Wait to finalize")
                pass
        elif self.mode == "predict":
            self.predict()
        else:
            # self.logger.info(self.mode)
            raise ValueError

    def pretrain(self):
        """
        pretrain the model
        :return:
        """
        raise NotImplementedError

    def train(self):
        """
        Main training loop
        :return:
        """
        raise NotImplementedError

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError

    def predict(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError

    def testing(self):
        """
        test the model
        :return:
        """
        raise NotImplementedError

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process.py the operator and the
        data loader
        :return:
        """
        # self.logger.info("Please wait while finalizing the operation.. Thank you")
        # self.logger.info("Saving checkpoint...")
        if self.save is True:
            self.save_checkpoint()
            self.save_results()
        # self.summary_writer.export_scalars_to_json(
        #     "{}all_scalars.json".format(self.config.summary_dir))
        # self.summary_writer.close()
        self.data_loader.finalize()
        return self.best_epoch, self.best_train_loss, self.best_val_perf
