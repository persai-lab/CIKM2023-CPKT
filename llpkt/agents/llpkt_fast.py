import numpy as np
import random
from tqdm import tqdm
import torch
from torch import nn
from torch.backends import cudnn
import torch.optim as optim
from sklearn import metrics

from llpkt.agents.base import BaseAgent
from llpkt.models.llpkt_fast import LLPKTFast
from llpkt.datasets.llpkt_fast import LLPKTFastDataset
from llpkt.datasets.llpkt_offline import LLPKTOfflineDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split

# should not remove import statements below, it;s being used seemingly.
from llpkt.dataloaders import *

cudnn.benchmark = True
from llpkt.utils.misc import *
# from ray import tune
import warnings

warnings.filterwarnings("ignore")


class LLPKTFastAgent(BaseAgent):
    def __init__(self, config):
        """initialize the agent with provided config dict which inherent from the base agent
        class"""
        super().__init__(config)

        # initialize the data_loader, which include preprocessing the data
        data_loader = globals()[config.data_loader]  # remember to import the dataloader
        self.data_loader = data_loader(config=config)
        self.mode = config.mode
        self.metric = config.metric
        self.offline_testing = config.offline_testing

        config.num_items = self.data_loader.num_items
        config.num_users = self.data_loader.num_users
        config.max_q_length = self.data_loader.max_q_length  # used in online training
        self.test_start_index = self.data_loader.start_test_index
        self.test_end_index = self.data_loader.end_test_index
        self.question_transition = self.data_loader.question_transition
        self.max_seq_len = self.data_loader.max_seq_len
        self.test_size = config.test_size
        self.hist_size = config.hist_size
        self.peek_size = config.peek_size
        self.num_items = config.num_items
        self.model = LLPKTFast(config)

        if self.metric == "rmse":
            self.mean_criterion = nn.MSELoss(reduction='mean')
            self.sum_criterion = nn.MSELoss(reduction='sum')
        elif self.metric == "auc":
            self.mean_criterion = nn.BCELoss(reduction='mean')
            self.sum_criterion = nn.BCELoss(reduction='sum')
        else:
            raise ValueError

        if self.config.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                eps=self.config.epsilon,
                weight_decay=self.config.weight_decay)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=0,
            min_lr=1e-6,
            factor=0.5,
            verbose=False
        )
        self.offline_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=0,
            min_lr=1e-6,
            factor=0.5,
            verbose=False
        )

        if self.cuda:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True

            self.device = torch.device("cuda")
            # torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.mean_criterion = self.mean_criterion.to(self.device)
            self.sum_criterion = self.sum_criterion.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            self.logger.info(
                strYellow("Torch Random Number: {}".format(torch.randint(0, 100, (3,)))))
            self.logger.info(
                strYellow("Numpy Random Number: {}".format(np.random.randint(0, 100, (3,)))))
            self.logger.info(strYellow("Random Number: {}".format(random.randint(0, 100))))
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            self.logger.info("Program will run on *****CPU*****\n")
            self.logger.info(strYellow("Random Number: {} ".format(torch.randint(0, 100, (3,)))))
            self.logger.info(
                strYellow("Numpy Random Number: {}".format(np.random.randint(0, 100, (3,)))))
            self.logger.info(strYellow("Random Number: {}".format(random.randint(0, 100))))

        # Model Loading from the latest checkpoint if not found start from scratch.
        # this loading should be after checking cuda
        # self.load_checkpoint(self.config.checkpoint_file)

        # for online training and testing
        self.test_dataloader = None
        self.start_test_index = self.data_loader.start_test_index
        self.end_test_index = self.data_loader.end_test_index
        self.current_test_index = 0
        self.train_perf_list = []
        self.val_perf_list = []
        self.sse_user_prob = self.config.sse_user_prob
        self.sse_item_prob = self.config.sse_item_prob
        self.sse_type = self.config.sse_type
        self.val_freq = self.config.val_freq
        self.user_outputs = {}

    def _reset_optimizer(self, learning_rate):
        if self.config.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                eps=self.config.epsilon,
                weight_decay=self.config.weight_decay)

    def _reset_scheduler(self):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=0,
            min_lr=1e-6,
            factor=0.5,
            verbose=False
        )
        self.offline_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=0,
            min_lr=1e-6,
            factor=0.5,
            verbose=False
        )

    def train(self):
        """
        :return:
        """

        self.train_offline()
        if self.offline_testing:
            pass
        else:
            # online training and testing with testing users' data
            self.pred_labels = []
            self.true_labels = []
            for i, attempt in enumerate(range(self.start_test_index, self.end_test_index,
                                              self.test_size)):
                self.current_epoch = 0
                self.current_test_index = attempt

                train_users = list(self.data_loader.data["train"]["user_records"].keys())
                train_users, val_users = train_test_split(train_users, test_size=0.25,
                                                          random_state=0)
                self.train_perf_list = []
                self.val_perf_list = []
                for epoch in range(1, self.config.max_epoch + 1):
                    self.load_data(self.current_test_index, self.config.batch_size,
                                   self.sse_user_prob, self.sse_item_prob, sse_type=self.sse_type,
                                   val_users=val_users, val_freq=self.val_freq)
                    self.train_one_epoch(mode="online")
                    self.current_epoch += 1
                    self.load_data(self.current_test_index, self.config.batch_size,
                                   sse_user_prob=0, sse_item_prob=0, sse_type=self.sse_type,
                                   val_users=val_users, val_freq=self.val_freq)
                    early_stop, pred_labels, true_labels = self.validate(mode="online")
                    if early_stop:
                        break

                # assert len(self.train_perf_list) == len(self.val_perf_list)
                # assert len(self.train_perf_list) != 0
                # lowest_val_error = min(self.val_perf_list)
                # idx = self.val_perf_list.index(lowest_val_error)
                # best_train_error = self.train_perf_list[idx]
                # print("lowest val error: {} and corresponding train error: {}".format(
                #     lowest_val_error, best_train_error
                # ))
                #
                # # train on all training data, early stop on best train error, then test
                # self._reset_optimizer(learning_rate=self.config.learning_rate * 0.1)
                # self._reset_scheduler()
                # for epoch in range(1, self.config.max_epoch + 1):
                #     self.load_data(self.current_test_index, self.config.batch_size,
                #                    self.sse_user_prob, self.sse_item_prob, sse_type=self.sse_type,
                #                    # val_users=val_users, val_freq=self.val_freq,
                #                    validation=False)
                #     self.train_one_epoch(mode="online")
                #     self.current_epoch += 1
                #     self.load_data(self.current_test_index, self.config.batch_size,
                #                    sse_user_prob=0, sse_item_prob=0, sse_type=self.sse_type,
                #                    # val_users=val_users, val_freq=self.val_freq,
                #                    validation=False)
                #     early_stop, pred_labels, true_labels = self.test(
                #         mode="online", best_train_error=best_train_error)
                #     if early_stop:
                #         break

                self.pred_labels.extend(pred_labels)
                self.true_labels.extend(true_labels)
                size, test_perf = self.finalize()
                # if self.mode == "hyperparameters":
                #     tune.report(t_perf=test_perf, t_size=size, t_index=attempt)
                # udpate memory slots
                _, _, _ = self.test(mode="online", update_memory=True)

                if self.metric == "rmse":
                    self.logger.info(strPurple("Overall Test Error (RMSE): {:.6f}, size: {}".format(
                        test_perf, size)))
                else:
                    self.logger.info(strPurple("Overall Test Error (AUC): {:.6f}, size: {}".format(
                        test_perf, size)))

                self._reset_optimizer(learning_rate=self.config.learning_rate)
                self._reset_scheduler()

        if self.mode != "hyperparameters":
            self.save_results()
            self.save_checkpoint()

    def train_one_epoch(self, mode="online"):
        self.model.train()
        self.logger.info("\n")

        if mode == "online":
            all_dataloader = self.all_online_dataloader
            self.logger.info(
                "Current Test Index:" + strGreen("[{}:{})".format(
                    self.current_test_index, self.current_test_index + self.test_size)) +
                ", Window Index:" + strGreen("[{}:{})".format(
                    max(0, self.current_test_index - self.hist_size),
                    self.current_test_index + self.test_size + self.peek_size)) +
                ", Train Epoch:" + strGreen("{}".format(self.current_epoch)))
            self.logger.info("learning rate: {}".format(self.optimizer.param_groups[0]['lr']))
        elif mode == "offline":
            all_dataloader = self.all_offline_dataloader
            self.logger.info("Train Epoch:" + strGreen("{}".format(self.current_epoch)))
            self.logger.info("learning rate: {}".format(self.optimizer.param_groups[0]['lr']))
        else:
            raise ValueError

        for idx, (questions, target_answers, user, interactions, train_target_mask,
                  val_target_mask, test_target_mask) in enumerate(tqdm(all_dataloader)):
            interactions = torch.tensor(interactions).to(self.device)
            questions = torch.tensor(questions).to(self.device)
            target_answers = torch.tensor(target_answers).to(self.device)
            train_target_mask = torch.tensor(train_target_mask).to(self.device)
            user = torch.tensor(user).to(self.device)
            self.optimizer.zero_grad()  # clear previous gradient
            output = self.model(questions, interactions, user, mode)

            train_label = torch.masked_select(target_answers, train_target_mask)
            train_output = torch.masked_select(output, train_target_mask)
            train_loss = self.sum_criterion(train_output.float(), train_label.float())
            train_loss.backward()  # compute the gradient
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()  # update the weight
            self.current_iteration += 1

    def validate(self, mode="online", update_memory=False):
        self.model.eval()
        self.train_loss = 0
        self.val_loss = 0
        self.test_loss = 0
        train_elements = 0
        val_elements = 0
        test_elements = 0

        pred_labels = []
        true_labels = []
        if mode == "online":
            all_dataloader = self.all_online_dataloader
        elif mode == "offline":
            all_dataloader = self.all_offline_dataloader
        else:
            raise ValueError

        with torch.no_grad():
            for idx, (questions, answers, user, interactions, train_target_mask,
                      val_target_mask, test_target_mask) in enumerate(tqdm(all_dataloader)):
                interactions = torch.tensor(interactions).to(self.device)
                questions = torch.tensor(questions).to(self.device)
                answers = torch.tensor(answers).to(self.device)
                train_target_mask = torch.tensor(train_target_mask).to(self.device)
                val_target_mask = torch.tensor(val_target_mask).to(self.device)
                test_target_mask = torch.tensor(test_target_mask).to(self.device)
                user = torch.tensor(user).to(self.device)
                output = self.model(questions, interactions, user, mode, update_memory)

                train_label = torch.masked_select(answers, train_target_mask)
                train_output = torch.masked_select(output, train_target_mask)
                train_loss = self.sum_criterion(train_output.float(), train_label.float())
                self.train_loss += train_loss.item()
                train_elements += train_target_mask.int().sum()

                val_label = torch.masked_select(answers, val_target_mask)
                val_output = torch.masked_select(output, val_target_mask)
                val_loss = self.sum_criterion(val_output.float(), val_label.float())
                self.val_loss += val_loss.item()
                val_elements += val_target_mask.int().sum()

                test_label = torch.masked_select(answers, test_target_mask)
                test_output = torch.masked_select(output, test_target_mask)
                test_loss = self.sum_criterion(test_output.float(), test_label.float())
                self.test_loss += test_loss.item()
                test_elements += test_target_mask.int().sum()

                pred_labels.extend(test_output.tolist())
                true_labels.extend(test_label.tolist())

        if self.metric == "rmse":
            self.train_loss = torch.sqrt(self.train_loss / train_elements)
            self.logger.info("Train Loss (RMSE): {:.6f}, size: {}".format(
                self.train_loss, train_elements))
            self.val_loss = torch.sqrt(self.val_loss / val_elements)
            self.logger.info("Val Loss (RMSE): {:.6f}, size: {}".format(
                self.val_loss, val_elements))
            self.test_loss = torch.sqrt(self.test_loss / test_elements)
            self.logger.info(strYellow("Test Loss (RMSE): {:.6f}, size: {}".format(
                self.test_loss, test_elements)))
        else:
            self.train_loss = self.train_loss / train_elements
            self.logger.info("Train Loss (BCE): {:.6f}, size: {}".format(
                self.train_loss, train_elements))
            self.val_loss = self.val_loss / val_elements
            self.logger.info("Val Loss (BCE): {:.6f}, size: {}".format(
                self.val_loss, val_elements))
            self.test_loss = self.test_loss / test_elements
            self.logger.info(strYellow("Test Loss (BCE): {:.6f}, size: {}".format(
                self.test_loss, test_elements)))

        if mode == "online":
            self.scheduler.step(self.val_loss)
        elif mode == "offline":
            self.offline_scheduler.step(self.val_loss)
        else:
            raise ValueError

        if len(self.val_perf_list) >= 5 and self.val_loss.cpu().numpy() > np.mean(
                self.val_perf_list[-5:]):
            self.train_perf_list.append(self.train_loss.item())
            self.val_perf_list.append(self.val_loss.item())
            return True, pred_labels, true_labels
        # elif len(self.val_perf_list) >= 1 and self.val_loss.cpu().numpy() < \
        #         self.train_loss.cpu().numpy():
        #     self.train_perf_list.append(self.train_loss.item())
        #     self.val_perf_list.append(self.val_loss.item())
        #     return True, pred_labels, true_labels
        else:
            self.train_perf_list.append(self.train_loss.item())
            self.val_perf_list.append(self.val_loss.item())
            return False, pred_labels, true_labels

    def test(self, mode="online", update_memory=False, best_train_error=0.):
        self.model.eval()
        self.train_loss = 0
        self.test_loss = 0
        train_elements = 0
        test_elements = 0

        pred_labels = []
        true_labels = []
        if mode == "online":
            all_dataloader = self.all_online_dataloader
        elif mode == "offline":
            all_dataloader = self.all_offline_dataloader
        else:
            raise ValueError

        with torch.no_grad():
            for idx, (questions, answers, user, interactions, train_target_mask,
                      val_target_mask, test_target_mask) in enumerate(tqdm(all_dataloader)):
                interactions = torch.tensor(interactions).to(self.device)
                questions = torch.tensor(questions).to(self.device)
                answers = torch.tensor(answers).to(self.device)
                train_target_mask = torch.tensor(train_target_mask).to(self.device)
                val_target_mask = torch.tensor(val_target_mask).to(self.device)
                test_target_mask = torch.tensor(test_target_mask).to(self.device)
                user = torch.tensor(user).to(self.device)
                output = self.model(questions, interactions, user, mode, update_memory)

                train_label = torch.masked_select(answers, train_target_mask)
                train_output = torch.masked_select(output, train_target_mask)
                train_loss = self.sum_criterion(train_output.float(), train_label.float())
                self.train_loss += train_loss.item()
                train_elements += train_target_mask.int().sum()

                print(user)
                print(test_target_mask)
                test_label = torch.masked_select(answers, test_target_mask)
                test_output = torch.masked_select(output, test_target_mask)
                test_loss = self.sum_criterion(test_output.float(), test_label.float())
                self.test_loss += test_loss.item()
                test_elements += test_target_mask.int().sum()

                pred_labels.extend(test_output.tolist())
                true_labels.extend(test_label.tolist())

                user = user.detach().numpy()

                for i in range(len(user)):
                    u = user[i]
                    u_mask = test_target_mask[i]
                    u_answer = answers[i]
                    u_output = output[i]
                    if u not in self.user_outputs:
                        self.user_outputs[u] = {}
                        self.user_outputs[u]["true"] = []
                        self.user_outputs[u]["pred"] = []
                    u_answer = torch.masked_select(u_answer, u_mask).detach().tolist()
                    u_output = torch.masked_select(u_output, u_mask).detach().tolist()
                    self.user_outputs[u]["true"] += u_answer
                    self.user_outputs[u]["pred"] += u_output


        if self.metric == "rmse":
            self.train_loss = torch.sqrt(self.train_loss / train_elements)
            self.logger.info("Train Loss (RMSE): {:.6f}, size: {}".format(
                self.train_loss, train_elements))
            self.test_loss = torch.sqrt(self.test_loss / test_elements)
            self.logger.info(strRed("Test Loss (RMSE): {:.6f}, size: {}".format(
                self.test_loss, test_elements)))

        else:
            self.train_loss = self.train_loss / train_elements
            self.logger.info("Train Loss (BCE): {:.6f}, size: {}".format(
                self.train_loss, train_elements))
            self.test_loss = self.test_loss / test_elements
            self.logger.info(strRed("Test Loss (BCE): {:.6f}, size: {}".format(
                self.test_loss, test_elements)))

        if mode == "online":
            self.scheduler.step(self.train_loss)
        elif mode == "offline":
            self.offline_scheduler.step(self.train_loss)
        else:
            raise ValueError

        if self.train_loss.cpu().numpy() <= best_train_error:
            return True, pred_labels, true_labels
        else:
            return False, pred_labels, true_labels

    def load_offline_data(self, batch_size, sse_user_prob=0., sse_item_prob=0., sse_type="ltm",
                          test=False):
        """
        user part of training users as validation users, and those validation data is not used for
        hyperparameter tuning, rather it is used for early stop
        :return:
        """
        data = self.data_loader.data
        train_users_records = data['train']['user_records'].copy()
        if test:
            # when testing, we combine train and val for training
            train_records = train_users_records
            val_records = {}
        else:
            train_users, val_users = train_test_split(list(train_users_records.keys()),
                                                      test_size=0.25,
                                                      random_state=self.config.seed)
            print("train users {}".format(len(train_users)))
            print("valid users {}".format(len(val_users)))
            # val_size = len(train_users_records) // 10
            train_records = {}
            val_records = {}
            # train_users = sorted(list(train_users_records.keys()))
            for user in train_users:
                train_records[user] = train_users_records[user]
            for user in val_users:
                val_records[user] = train_users_records[user]
        test_records = data['test']['user_records'].copy()

        all_users_data = LLPKTOfflineDataset(train_records, val_records, test_records,
                                             self.num_items,
                                             max_seq_len=self.max_seq_len,
                                             test_start_index=self.test_start_index,
                                             test_end_index=self.test_end_index,
                                             question_transition=self.question_transition,
                                             metric=self.metric,
                                             sse_user_prob=sse_user_prob,
                                             sse_item_prob=sse_item_prob,
                                             sse_type=sse_type,
                                             seed=self.config.seed)
        init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.config.shuffle,
            'collate_fn': default_collate,
            'num_workers': self.config.num_workers
        }
        self.all_offline_dataloader = DataLoader(all_users_data, **init_kwargs)

    def load_data(self, current_test_index, batch_size, sse_user_prob=0.,
                  sse_item_prob=0., sse_type="ltm", val_users=None, val_freq=2):
        """
        user part of training users as validation users, and those validation data is not used for
        hyperparameter tuning, rather it is used for early stop
        :return:
        """
        data = self.data_loader.data
        max_q_length = self.data_loader.max_q_length
        train_user_records = data['train']['user_records']
        test_user_records = data['test']['user_records']
        all_users_data = LLPKTFastDataset(train_user_records, test_user_records, self.num_items,
                                          test_start_index=current_test_index,
                                          test_size=self.test_size,
                                          hist_size=self.hist_size,
                                          peek_size=self.peek_size,
                                          question_transition=self.question_transition,
                                          metric=self.metric,
                                          sse_user_prob=sse_user_prob,
                                          sse_item_prob=sse_item_prob,
                                          sse_type=sse_type,
                                          val_users=val_users,
                                          val_freq=val_freq,
                                          seed=self.config.seed)
        init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.config.shuffle,
            'collate_fn': default_collate,
            'num_workers': self.config.num_workers
        }
        self.all_online_dataloader = DataLoader(all_users_data, **init_kwargs)

    def train_offline(self):
        if self.sse_user_prob > 0 or self.sse_item_prob > 0.:
            # # offline training with training users' data
            # train and val to find best training error
            self.current_epoch = 0
            self.train_perf_list = []
            self.val_perf_list = []
            self.pred_labels = []
            self.true_labels = []
            for epoch in range(1, self.config.max_offline_epoch + 1):
                self.current_epoch = epoch
                self.load_offline_data(self.config.batch_size,
                                       sse_user_prob=self.sse_user_prob,
                                       sse_item_prob=self.sse_item_prob)
                self.train_one_epoch(mode="offline")
                self.current_epoch += 1
                self.load_offline_data(self.config.batch_size)
                early_stop, pred_labels, true_labels = self.validate(mode="offline")
                if early_stop:
                    break
                # _, _, _ = self.validate(mode="offline")
            # assert len(self.train_perf_list) == len(self.val_perf_list)
            # assert len(self.train_perf_list) != 0
            # lowest_val_error = min(self.val_perf_list)
            # idx = self.val_perf_list.index(lowest_val_error)
            # best_train_error = self.train_perf_list[idx]
            # print("lowest val error: {} and corresponding train error: {}".format(
            #     lowest_val_error, best_train_error
            # ))
            #
            # # train on all training data, early stop on best train error, then test
            # self._reset_optimizer(learning_rate=self.config.learning_rate)
            # self._reset_scheduler()
            # for epoch in range(1, self.config.max_offline_epoch + 1):
            #     self.current_epoch = epoch
            #     self.load_offline_data(self.config.batch_size,
            #                            sse_user_prob=self.sse_user_prob,
            #                            sse_item_prob=self.sse_item_prob,
            #                            test=True)
            #     self.train_one_epoch(mode="offline")
            #     self.current_epoch += 1
            #     self.load_offline_data(self.config.batch_size, test=True)
            #     early_stop, pred_labels, true_labels = self.test(mode="offline",
            #                                                      best_train_error=best_train_error)
            #     if early_stop:
            #         break

            self.pred_labels.extend(pred_labels)
            self.true_labels.extend(true_labels)
            size, test_perf = self.finalize_offline()
            if self.metric == "rmse":
                self.logger.info(
                    strGreen("Overall Offline Test Error (RMSE): {:.6f}, size: {}".format(
                        test_perf, size)))
            else:
                self.logger.info(
                    strGreen("Overall Offline Test Error (AUC): {:.6f}, size: {}".format(
                        test_perf, size)))
            if self.mode == "hyperparameters":
                # tune.report(t_perf=test_perf, t_size=size)
                pass
            # self.optimizer.param_groups[0]['lr'] = self.config.learning_rate
            self.config.learning_rate = self.config.learning_rate * 0.01
            self._reset_optimizer(self.config.learning_rate)
            self._reset_scheduler()

    def finalize_offline(self):
        size = len(self.pred_labels)
        if self.metric == "rmse":
            perf = np.sqrt(
                metrics.mean_squared_error(self.true_labels, self.pred_labels)
            )
        elif self.metric == "auc":
            perf = metrics.roc_auc_score(self.true_labels, self.pred_labels)
        else:
            raise ValueError
        # self.save_checkpoint()
        return size, perf

    def finalize(self):
        size = len(self.pred_labels)
        if self.metric == "rmse":
            perf = np.sqrt(metrics.mean_squared_error(self.true_labels, self.pred_labels))
        elif self.metric == "auc":
            perf = metrics.roc_auc_score(self.true_labels, self.pred_labels)
        else:
            raise ValueError
        return size, perf
