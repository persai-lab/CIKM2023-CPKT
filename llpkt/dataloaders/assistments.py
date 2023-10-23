import os
import pickle
import numpy as np

from llpkt.dataloaders.base import BaseDataLoader
from pathlib import Path


class ASSISTmentsDataLoader(BaseDataLoader):
    def __init__(self, config):
        """
        initialize the dataset, train_loader, test_loader
        :param config:
        """
        super().__init__(config)
        self.data_name = config["data_name"]
        data_dir = "../data/"
        data_path = os.path.join(data_dir, "ASSISTments", f"{self.data_name}.pkl")
        self.data = pickle.load(open(data_path, "rb"))
        self.num_items = self.data["num_skills"]
        self.num_users = self.data["num_users"]
        print("num users: {}".format(self.num_users))
        print("num items: {}".format(self.num_items))
        self.question_transition = self.data["transition"]
        self.start_test_index = config.test_start_index
        # self.end_test_index = 1000
        self.end_test_index = self.data["test"]["max_q_length"]
        self.max_q_length = max(self.data["train"]["max_q_length"],
                                self.data["test"]["max_q_length"])

    def finalize(self):
        pass
