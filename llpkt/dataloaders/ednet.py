import os
import pickle
import numpy as np

from llpkt.dataloaders.base import BaseDataLoader
from pathlib import Path


class EdNetDataLoader(BaseDataLoader):
    def __init__(self, config):
        """
        initialize the dataset, train_loader, test_loader
        :param config:
        """
        super().__init__(config)
        # working_dir = config["working_dir"]
        self.data_name = config["data_name"]
        print("*****************************************************", Path.cwd())
        # data_path = os.path.abspath(f"../data/MORF/{self.data_name}.pkl")
        # data_dir = os.path.abspath("{}/../data/".format(working_dir))
        data_dir = "../data/"
        data_path = os.path.join(data_dir, "EdNet", f"{self.data_name}.pkl")
        self.data = pickle.load(open(data_path, "rb"))
        self.num_items = self.data["num_questions"]
        self.num_nonassessed_items = self.data["num_lectures"]
        self.num_users = self.data["num_users"]
        print("num users: {}".format(self.num_users))
        print("num items: {}".format(self.num_items))
        self.question_transition = self.data["question_transition"]
        self.start_test_index = config.test_start_index
        self.end_test_index = 1000
        # self.end_test_index = self.data["test"]["max_q_length"]
        self.max_q_length = max(self.data["train"]["max_q_length"],
                                self.data["test"]["max_q_length"])

    def finalize(self):
        pass
