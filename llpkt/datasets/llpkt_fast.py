import numpy as np
from torch.utils.data import Dataset
import random
from llpkt.datasets.transforms import SlidingWindow, Padding
import torch


class LLPKTFastDataset(Dataset):
    """
    prepare the data for data loader, including truncating long sequence and padding
    """

    def __init__(self, train_user_records, test_user_records, num_items, test_start_index,
                 test_size, hist_size, peek_size, question_transition, metric="auc",
                 sse_user_prob=0., sse_item_prob=0., sse_type="ltm",
                 val_users=None, val_freq=5, seed=1024):
        """
        param max_seq_len: used to truncate seq. greater than max_seq_len
        :param max_subseq_len: used to truncate the lecture seq.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.train_user_records = train_user_records
        self.test_user_records = test_user_records
        self.num_items = num_items
        self.test_start_index = test_start_index
        self.test_size = test_size
        self.hist_size = hist_size
        self.peek_size = peek_size
        self.window_size = hist_size + test_size + peek_size
        self.val_users = val_users
        self.val_freq = val_freq

        self.metric = metric
        self.sse_user_prob = sse_user_prob
        self.sse_item_prob = sse_item_prob
        self.sse_type = sse_type
        self.question_transition = question_transition
        self.user_id_mapping = {}
        all_user_records = {**train_user_records, **test_user_records}
        # all_user_records = {**test_user_records}
        outputs = self._transform(all_user_records)
        self.q_data, self.a_data = outputs[:2]
        self.train_mask_data, self.val_mask_data, self.test_mask_data = outputs[2:]

        # print("train samples.: {}".format(self.q_data.shape))
        assert len(self.q_data) == len(self.a_data)
        self.length = len(self.q_data)

    def __len__(self):
        """
        :return: the number of training samples rather than training users
        """
        return self.length

    def __getitem__(self, idx):
        """
        for online learning version
        idx: sample index
        """
        questions = self.q_data[idx]
        answers = self.a_data[idx]
        train_target_mask = self.train_mask_data[idx]
        val_target_mask = self.val_mask_data[idx]
        test_target_mask = self.test_mask_data[idx]
        assert len(questions) == len(answers)

        user = self.user_id_mapping[idx]
        if self.metric == "rmse":
            interactions = []
            for i in range(len(questions)):
                q = questions[i]
                if q != 0:  # if q == 0, that is padding
                    if np.random.random() < self.sse_item_prob:
                        if self.sse_type == "random":
                            q = np.random.randint(1, self.num_items + 1)
                            questions[i] = q
                        else:
                            next_items = self.question_transition[q]
                            if len(next_items) == 0:
                                continue
                            j = np.random.choice(range(len(next_items)))
                            q, a = next_items[j]
                            questions[i] = q
                    interactions.append([q, answers[i]])
                else:
                    interactions.append([0, 0])
            interactions = np.array(interactions, dtype=float)
        else:
            interactions = np.zeros_like(questions, dtype=int)
            for i in range(len(questions)):
                q = questions[i]
                if q != 0:
                    if np.random.random() < self.sse_item_prob:
                        if self.sse_type == "random":
                            q = np.random.randint(1, self.num_items + 1)
                            questions[i] = q
                        else:
                            next_items = self.question_transition[q]
                            if len(next_items) == 0:
                                continue
                            j = np.random.choice(range(len(next_items)))
                            q, a = next_items[j]
                            questions[i] = q
                    interactions[i] = q + answers[i] * self.num_items
                else:
                    interactions[i] = 0

        if np.random.random() < self.sse_user_prob:
            user = np.random.choice(list(self.user_id_mapping.values()))

        return questions, answers, user, interactions, train_target_mask, val_target_mask, \
               test_target_mask

    def _transform(self, all_user_records):
        """
        transform the data into feasible input of model,
        truncate the seq. if it is too long and
        pad the seq. with 0s if it is too short

        we don't train the seq if current_test_index is > q_len
        """
        self.user_id_mapping = {}
        q_data = []
        a_data = []
        train_mask_data = []
        val_mask_data = []
        test_mask_data = []
        window_padding = Padding(self.window_size, side='right', fillvalue=0)
        mask_padding = Padding(self.window_size, side='right', fillvalue=False)
        for user in sorted(list(all_user_records.keys())):
            q_len = len(all_user_records[user]['q'])
            q_list = all_user_records[user]['q'].copy()
            a_list = all_user_records[user]['a'].copy()
            assert len(q_list) == len(a_list)

            # initialize the masks
            full_train_mask = [True] * q_len
            full_val_mask = [False] * q_len
            full_test_mask = [False] * q_len
            remain = user % self.val_freq
            for i in range(remain, q_len, self.val_freq):
                full_train_mask[i] = False
                full_val_mask[i] = True

            if q_len <= self.test_start_index - self.hist_size:
                continue  # we need skip those samples to correctly update value matrix
                # questions = q_list[-self.window_size:]
                # answers = a_list[-self.window_size:]
                # train_mask = full_train_mask[-self.window_size:]
                # val_mask = full_val_mask[-self.window_size:]
                # test_mask = full_test_mask[-self.window_size:]
            else:
                # adjust the mask accordingly
                if self.test_start_index < self.hist_size:
                    questions = q_list[:self.test_start_index + self.test_size + self.peek_size]
                    answers = a_list[:self.test_start_index + self.test_size + self.peek_size]
                    if user in self.test_user_records:
                        full_train_mask = np.array(full_train_mask)
                        full_val_mask = np.array(full_val_mask)
                        full_test_mask = np.array(full_test_mask)
                        full_train_mask[self.test_start_index:] = False
                        full_val_mask[self.test_start_index:] = False
                        full_test_mask[self.test_start_index:
                                       self.test_start_index + self.test_size] = True
                        full_train_mask = list(full_train_mask)
                        full_val_mask = list(full_val_mask)
                        full_test_mask = list(full_test_mask)
                    else:
                        if user in self.val_users:
                            full_train_mask = np.array(full_train_mask)
                            full_val_mask = np.array(full_val_mask)
                            full_test_mask = np.array(full_test_mask)
                            full_train_mask[:self.test_start_index] = True
                            full_train_mask[self.test_start_index:
                                            self.test_start_index + self.test_size] = False
                            full_val_mask[:self.test_start_index] = False
                            full_val_mask[self.test_start_index:
                                          self.test_start_index + self.test_size] = True
                            full_train_mask = list(full_train_mask)
                            full_val_mask = list(full_val_mask)
                            full_test_mask = list(full_test_mask)
                        # else:
                        #     full_train_mask = np.array(full_train_mask)
                        #     full_val_mask = np.array(full_val_mask)
                        #     full_test_mask = np.array(full_test_mask)
                        #     full_train_mask[:] = True
                        #     full_val_mask[:] = False
                        #     full_train_mask = list(full_train_mask)
                        #     full_val_mask = list(full_val_mask)
                        #     full_test_mask = list(full_test_mask)

                    train_mask = full_train_mask[
                                 :self.test_start_index + self.test_size + self.peek_size]
                    val_mask = full_val_mask[
                               :self.test_start_index + self.test_size + self.peek_size]
                    test_mask = full_test_mask[
                                :self.test_start_index + self.test_size + self.peek_size]
                else:
                    questions = q_list[self.test_start_index - self.hist_size:
                                       self.test_start_index + self.test_size]
                    answers = a_list[self.test_start_index - self.hist_size:
                                     self.test_start_index + self.test_size]
                    if user in self.test_user_records:
                        full_train_mask = np.array(full_train_mask)
                        full_val_mask = np.array(full_val_mask)
                        full_test_mask = np.array(full_test_mask)
                        full_train_mask[self.test_start_index:] = False
                        full_val_mask[self.test_start_index:] = False
                        full_test_mask[self.test_start_index:
                                       self.test_start_index + self.test_size] = True
                        full_train_mask = list(full_train_mask)
                        full_val_mask = list(full_val_mask)
                        full_test_mask = list(full_test_mask)
                    else:
                        if user in self.val_users:
                            full_train_mask = np.array(full_train_mask)
                            full_val_mask = np.array(full_val_mask)
                            full_test_mask = np.array(full_test_mask)
                            full_train_mask[:self.test_start_index] = True
                            full_train_mask[self.test_start_index:
                                            self.test_start_index + self.test_size] = False
                            full_val_mask[:self.test_start_index] = False
                            full_val_mask[self.test_start_index:
                                          self.test_start_index + self.test_size] = True
                            full_train_mask = list(full_train_mask)
                            full_val_mask = list(full_val_mask)
                            full_test_mask = list(full_test_mask)
                        # else:
                        #     full_train_mask = np.array(full_train_mask)
                        #     full_val_mask = np.array(full_val_mask)
                        #     full_test_mask = np.array(full_test_mask)
                        #     full_train_mask[:] = True
                        #     full_val_mask[:] = False
                        #     full_train_mask = list(full_train_mask)
                        #     full_val_mask = list(full_val_mask)
                        #     full_test_mask = list(full_test_mask)
                    train_mask = full_train_mask[self.test_start_index - self.hist_size:
                                                 self.test_start_index + self.test_size + self.peek_size]
                    val_mask = full_val_mask[self.test_start_index - self.hist_size:
                                             self.test_start_index + self.test_size + self.peek_size]
                    test_mask = full_test_mask[self.test_start_index - self.hist_size:
                                               self.test_start_index + self.test_size + self.peek_size]

            sample = {"q": questions, "a": answers}
            output = window_padding(sample)
            questions = output['q']
            answers = output['a']
            mask_sample = {"train": train_mask, "val": val_mask, "test": test_mask}
            mask_output = mask_padding(mask_sample)
            train_mask = mask_output["train"]
            val_mask = mask_output["val"]
            test_mask = mask_output["test"]

            id = len(q_data)
            self.user_id_mapping[id] = user
            q_data.append(questions)
            a_data.append(answers)
            train_mask_data.append(train_mask)
            val_mask_data.append(val_mask)
            test_mask_data.append(test_mask)

        return np.array(q_data), np.array(a_data), np.array(train_mask_data), \
               np.array(val_mask_data), np.array(test_mask_data)


if __name__ == '__main__':
    from transforms import SlidingWindow, Padding
    from torch.utils.data import DataLoader
    from torch.utils.data.dataloader import default_collate

    train_user_records = {}
    transition = {}
    num_items = 10
    num_nonassessed_items = 10
    train_user_records = {
        1: {'q': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'a': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'l': [[11], [11, 12], [13], [], [13, 14, 15], [], [], [], [], [13, 18, 17, 19], [], [],
                  [], [], [], [], [], [18, 19, 20], [], []]},
        2: {'q': [2, 3, 4, 5, 6, 7, 8, 9, 10, 1],
            'a': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'l': [[12], [13], [], [12, 14, 15], [13, 14], [14, 15, 17], [16, 12], [], [], []]},
    }
    test_user_records = {
        3: {'q': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'a': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'l': [[11], [11, 12], [13], [], [13, 14, 15], [], [], [], [], [13, 18, 17, 19], [], [],
                  [], [], [], [], [], [18, 19, 20], [], []]},
        4: {'q': [2, 3, 4, 5, 6, 7, 8, 9, 10, 1],
            'a': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'l': [[12], [13], [], [12, 14, 15], [13, 14], [14, 15, 17], [16, 12], [], [], []]},
    }

    # test_index = 9
    # test_size=5
    # hist_size = 3

    test_index = 14
    test_size = 5
    hist_size = 3

    test_users_data = LLPKTFastDataset(train_user_records, test_user_records, num_items,
                                       test_start_index=test_index,
                                       test_size=test_size,
                                       hist_size=hist_size,
                                       question_transition=transition,
                                       metric="auc",
                                       sse_user_prob=0.,
                                       sse_item_prob=0.,
                                       sse_type="random",
                                       val_freq=2,
                                       seed=1024)
    print(test_users_data)
    init_kwargs = {
        'batch_size': 1,
        'shuffle': True,
        'collate_fn': default_collate,
        'num_workers': 1
    }
    test_users_dataloader = DataLoader(test_users_data, **init_kwargs)
    for idx, (questions, target_answers, user, interactions, train_target_mask,
              val_target_mask, test_target_mask) in enumerate(test_users_dataloader):
        print('user', user)
        print('question', questions)
        print('answer', target_answers)
        print('interaction', interactions)
        print('train_mask', train_target_mask)
        print('val_mask', val_target_mask)
        print('test_mask', test_target_mask)
        print()

    # test_users_data = LLPKTMultiTypeDataset(train_user_records, test_user_records, num_items,
    #                                         num_nonassessed_items,
    #                                         current_test_index=test_index,
    #                                         window_size=window_size,
    #                                         max_seq_len=10,
    #                                         max_subseq_len=2,
    #                                         transition_dict=transition,
    #                                         metric="auc",
    #                                         sse_prob=0.,
    #                                         sse_type="random",
    #                                         seed=1024)
    # print(test_users_data)
    # init_kwargs = {
    #     'batch_size': 1,
    #     'shuffle': True,
    #     'collate_fn': default_collate,
    #     'num_workers': 1
    # }
    # test_users_dataloader = DataLoader(test_users_data, **init_kwargs)
    # for idx, (questions, target_answers, lectures, user, interactions, train_target_mask,
    #           val_target_mask, test_target_mask) in enumerate(test_users_dataloader):
    #     print('user', user)
    #     print('question', questions)
    #     print('answer', target_answers)
    #     print('lecture', lectures)
    #     print('interaction', interactions)
    #     print('train_mask', train_target_mask)
    #     print('val_mask', val_target_mask)
    #     print('test_mask', test_target_mask)
    #     print()
