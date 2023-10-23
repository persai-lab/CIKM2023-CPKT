import numpy as np
from torch.utils.data import Dataset
import random
# from llpkt.datasets.transforms import SlidingWindow, Padding
import torch


class LLPKTDataset(Dataset):
    """
    prepare the data for data loader, including truncating long sequence and padding
    """

    def __init__(self, train_user_records, test_user_records, num_items, current_test_index,
                 window_size, max_seq_len, question_transition, metric="auc",
                 sse_prob=0., sse_type="ltm", seed=1024):
        """
        param max_seq_len: used to truncate seq. greater than max_seq_len
        :param max_subseq_len: used to truncate the lecture seq.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.train_user_records = train_user_records
        self.test_user_records = test_user_records
        self.num_items = num_items
        self.current_test_index = current_test_index
        self.window_size = window_size
        self.max_seq_len = max_seq_len
        self.metric = metric
        self.sse_prob = sse_prob
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
                    if np.random.random() < self.sse_prob:
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
                    if np.random.random() < self.sse_prob:
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

        if np.random.random() < 0.5:
            user = np.random.choice(list(self.user_id_mapping.keys()))

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
            if q_len <= self.current_test_index:
                # continue
                questions = q_list[-self.window_size:]
                answers = a_list[-self.window_size:]
                train_mask = list((np.array(questions) != 0))
                val_mask = [False] * len(questions)
                test_mask = [False] * len(questions)
            else:
                if self.current_test_index + 1 >= self.window_size:
                    questions = q_list[self.current_test_index + 1 - self.window_size:
                                       self.current_test_index + 1]
                    answers = a_list[self.current_test_index + 1 - self.window_size:
                                     self.current_test_index + 1]
                    if user in self.train_user_records:
                        train_mask = list((np.array(questions) != 0))
                        val_mask = [False] * len(questions)
                        train_mask[-1] = False
                        val_mask[-1] = True
                        test_mask = [False] * self.window_size
                    else:
                        train_mask = list((np.array(questions) != 0))
                        val_mask = [False] * len(questions)
                        for i in range(2, len(questions) - 1, 10):
                            if train_mask[i]:
                                train_mask[i] = False
                                val_mask[i] = True
                        train_mask[-1] = False
                        val_mask[-1] = False
                        test_mask = [False] * self.window_size
                        test_mask[-1] = True
                else:
                    if user in self.train_user_records:
                        questions = q_list[:self.window_size]
                        answers = a_list[:self.window_size]
                        train_mask = list((np.array(questions) != 0))
                        val_mask = [False] * len(questions)
                        train_mask[-1] = False
                        val_mask[-1] = True
                        test_mask = [False] * self.window_size
                    else:
                        questions = q_list[:self.current_test_index + 1]
                        answers = a_list[:self.current_test_index + 1]
                        train_mask = list((np.array(questions) != 0))
                        val_mask = [False] * len(questions)
                        for i in range(2, len(questions) - 1, 10):
                            if train_mask[i]:
                                train_mask[i] = False
                                val_mask[i] = True
                        train_mask[-1] = False
                        val_mask[-1] = False
                        test_mask = [False] * len(questions)
                        test_mask[-1] = True

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


# class LLPKTDataset(Dataset):
#     """
#     the dataloader with best performance so far
#     prepare the data for data loader, including truncating long sequence and padding
#     """
#
#     def __init__(self, train_user_records, test_user_records, num_items, current_test_index,
#                  window_size, max_seq_len, question_transition, metric="auc",
#                  sse_prob=0., sse_type="ltm", seed=1024):
#         """
#         param max_seq_len: used to truncate seq. greater than max_seq_len
#         :param max_subseq_len: used to truncate the lecture seq.
#         """
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         self.train_user_records = train_user_records
#         self.test_user_records = test_user_records
#         self.num_items = num_items
#         self.current_test_index = current_test_index
#         self.window_size = window_size
#         self.max_seq_len = max_seq_len
#         self.metric = metric
#         self.sse_prob = sse_prob
#         self.sse_type = sse_type
#         self.question_transition = question_transition
#         self.user_id_mapping = {}
#         all_user_records = {**train_user_records, **test_user_records}
#         # all_user_records = {**test_user_records}
#         outputs = self._transform(all_user_records)
#         self.q_data, self.a_data = outputs[:2]
#
#         # print("train samples.: {}".format(self.q_data.shape))
#         assert len(self.q_data) == len(self.a_data)
#         self.length = len(self.q_data)
#
#     def __len__(self):
#         """
#         :return: the number of training samples rather than training users
#         """
#         return self.length
#
#     def __getitem__(self, idx):
#         """
#         for online learning version
#         idx: sample index
#         """
#         questions = self.q_data[idx]
#         answers = self.a_data[idx]
#         assert len(questions) == len(answers)
#
#         user = self.user_id_mapping[idx]
#         if self.metric == "rmse":
#             interactions = []
#             for i in range(len(questions)):
#                 q = questions[i]
#                 if q != 0:  # if q == 0, that is padding
#                     if np.random.random() < self.sse_prob:
#                         if self.sse_type == "random":
#                             q = np.random.randint(1, self.num_items + 1)
#                             questions[i] = q
#                         else:
#                             next_items = self.question_transition[q]
#                             if len(next_items) == 0:
#                                 continue
#                             j = np.random.choice(range(len(next_items)))
#                             q, a = next_items[j]
#                             questions[i] = q
#                     interactions.append([q, answers[i]])
#                 else:
#                     interactions.append([0, 0])
#             interactions = np.array(interactions, dtype=float)
#         else:
#             interactions = np.zeros_like(questions, dtype=int)
#             for i in range(len(questions)):
#                 q = questions[i]
#                 if q != 0:
#                     if np.random.random() < self.sse_prob:
#                         if self.sse_type == "random":
#                             q = np.random.randint(1, self.num_items + 1)
#                             questions[i] = q
#                         else:
#                             next_items = self.question_transition[q]
#                             if len(next_items) == 0:
#                                 continue
#                             j = np.random.choice(range(len(next_items)))
#                             q, a = next_items[j]
#                             questions[i] = q
#                     interactions[i] = q + answers[i] * self.num_items
#                 else:
#                     interactions[i] = 0
#
#         questions = np.insert(questions, 0, 0)
#         answers = np.insert(answers, 0, 0)
#         if self.metric == "rmse":
#             interactions = np.insert(interactions, 0, [0, 0], axis=0)
#         else:
#             interactions = np.insert(interactions, 0, 0)
#
#         assert self.window_size <= len(questions)
#
#         if user in self.train_user_records:
#             q_len = len(self.train_user_records[user]['q'])
#             # note that len(questions) >= q_len + 1
#             if q_len < self.window_size:
#                 questions = questions[:self.window_size]
#                 answers = answers[:self.window_size]
#                 interactions = interactions[:self.window_size]
#                 target_mask = (questions != 0)
#                 train_target_mask = target_mask.copy()
#                 val_target_mask = np.array([False] * self.window_size)
#                 train_target_mask[q_len] = False
#                 val_target_mask[q_len] = True
#                 test_target_mask = np.array([False] * self.window_size)
#             else:
#                 if self.current_test_index < (self.window_size // 2):
#                     questions = questions[:self.window_size]
#                     answers = answers[:self.window_size]
#                     interactions = interactions[:self.window_size]
#                     target_mask = (questions != 0)
#                     train_target_mask = target_mask.copy()
#                     val_target_mask = np.array([False] * self.window_size)
#                     train_target_mask[-1] = False
#                     val_target_mask[-1] = True
#                     test_target_mask = np.array([False] * self.window_size)
#                 elif self.current_test_index > (q_len - self.window_size // 2):
#                     # q_len is last index of questions
#                     questions = questions[q_len - self.window_size + 1: q_len + 1]
#                     answers = answers[q_len - self.window_size + 1: q_len + 1]
#                     interactions = interactions[q_len - self.window_size + 1: q_len + 1]
#                     target_mask = (questions != 0)
#                     train_target_mask = target_mask.copy()
#                     val_target_mask = np.array([False] * self.window_size)
#                     train_target_mask[-1] = False
#                     val_target_mask[-1] = True
#                     test_target_mask = np.array([False] * self.window_size)
#                 else:
#                     start_index = self.current_test_index - (self.window_size // 2) + 1
#                     questions = questions[start_index: start_index + self.window_size]
#                     answers = answers[start_index: start_index + self.window_size]
#                     interactions = interactions[start_index: start_index + self.window_size]
#                     target_mask = (questions != 0)
#                     train_target_mask = target_mask.copy()
#                     val_target_mask = np.array([False] * self.window_size)
#                     train_target_mask[-1] = False
#                     val_target_mask[-1] = True
#                     test_target_mask = np.array([False] * self.window_size)
#         elif user in self.test_user_records:
#             q_len = len(self.test_user_records[user]['q'])
#             if q_len < self.window_size:
#                 questions = questions[:self.window_size]
#                 answers = answers[:self.window_size]
#                 interactions = interactions[:self.window_size]
#                 if self.current_test_index >= q_len:
#                     # this test user's all data become training data
#                     target_mask = (questions != 0)
#                     train_target_mask = target_mask.copy()
#                     val_target_mask = np.array([False] * self.window_size)
#                     train_target_mask[q_len] = False
#                     val_target_mask[q_len] = True
#                     test_target_mask = np.array([False] * self.window_size)
#                 else:
#                     train_target_mask = np.array([False] * self.window_size)
#                     for i in range(1, self.current_test_index + 1, 1):
#                         # because we insert 0 at the beginning, so iterate up to current_test_index
#                         # questions[self.current_test_index + 1] is the test data point
#                         train_target_mask[i] = True
#                     val_target_mask = np.array([False] * self.window_size)
#                     test_target_mask = np.array([False] * self.window_size)
#                     test_target_mask[self.current_test_index + 1] = True
#             else:
#                 # window_size <= q_len
#                 if self.current_test_index >= q_len:
#                     # this test user's last window_size number of data become training data
#                     questions = questions[q_len - self.window_size + 1: q_len + 1]
#                     answers = answers[q_len - self.window_size + 1: q_len + 1]
#                     interactions = interactions[q_len - self.window_size + 1: q_len + 1]
#                     target_mask = (questions != 0)
#                     train_target_mask = target_mask.copy()
#                     val_target_mask = np.array([False] * self.window_size)
#                     train_target_mask[-1] = False
#                     val_target_mask[-1] = True
#                     test_target_mask = np.array([False] * self.window_size)
#                 else:
#                     # when current_test_index < q_len
#                     if self.current_test_index + 1 < self.window_size:
#                         questions = questions[:self.window_size]
#                         answers = answers[:self.window_size]
#                         interactions = interactions[:self.window_size]
#                         train_target_mask = np.array([False] * self.window_size)
#                         for i in range(1, self.current_test_index + 1, 1):
#                             train_target_mask[i] = True
#                         val_target_mask = np.array([False] * self.window_size)
#                         test_target_mask = np.array([False] * self.window_size)
#                         test_target_mask[self.current_test_index + 1] = True
#                     else:
#                         start_index = self.current_test_index + 2 - self.window_size
#                         questions = questions[start_index: self.current_test_index + 2]
#                         answers = answers[start_index: self.current_test_index + 2]
#                         interactions = interactions[start_index: self.current_test_index + 2]
#                         train_target_mask = np.array([False] * self.window_size)
#                         for i in range(0, self.window_size-1, 1):
#                             train_target_mask[i] = True
#                         val_target_mask = np.array([False] * self.window_size)
#                         test_target_mask = np.array([False] * self.window_size)
#                         test_target_mask[-1] = True
#         else:
#             raise ValueError
#
#         # if np.random.random() < self.sse_prob:
#         #     user = np.random.choice(list(self.user_id_mapping.keys()))
#         # print("idx: {}, questions {}".format(idx, questions))
#         # print("answers {}".format(answers))
#         # print("interactions {}".format(interactions))
#         # print("lectures {}".format(lectures))
#         # print("train_target_masks {}".format(train_target_mask))
#         # print("val_target_masks {}".format(val_target_mask))
#         # print("test_target_masks {}".format(test_target_mask))
#         # return questions, answers, user, interactions, lectures, train_target_mask, \
#         #        val_target_mask, test_target_mask
#         return questions, answers, user, interactions, train_target_mask, \
#                val_target_mask, test_target_mask
#
#     def _transform(self, all_user_records):
#         """
#         transform the data into feasible input of model,
#         truncate the seq. if it is too long and
#         pad the seq. with 0s if it is too short
#         """
#         q_data = []
#         a_data = []
#         padding = Padding(self.max_seq_len, side='right', fillvalue=0)
#         for user in sorted(list(all_user_records.keys())):
#             q_list = all_user_records[user]['q']
#             a_list = all_user_records[user]['a']
#             assert len(q_list) == len(a_list)
#             sample = {"q": q_list, "a": a_list}
#             output = padding(sample)  # output['q'] is 1d list
#             id = len(q_data)
#             self.user_id_mapping[id] = user
#             q_data.append(output['q'])
#             a_data.append(output['a'])
#         users = list(sorted(all_user_records.keys()))
#         assert users == [i for i in range(1, len(all_user_records) + 1)]
#         return np.array(q_data), np.array(a_data)


class LLPKTMultiTypeDataset(Dataset):
    """
    prepare the data for data loader, including truncating long sequence and padding
    """

    def __init__(self, train_user_records, test_user_records, num_items, num_nonassessed_items,
                 current_test_index, window_size, max_seq_len, max_subseq_len, question_transition,
                 lecture_transition, multitype_transition, metric="auc", sse_prob=0.,
                 sse_type="ltm", seed=1024):
        """
        param max_seq_len: used to truncate seq. greater than max_seq_len
        :param max_subseq_len: used to truncate the lecture seq.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.train_user_records = train_user_records
        self.test_user_records = test_user_records
        self.num_items = num_items
        self.num_nonassessed_items = num_nonassessed_items
        self.current_test_index = current_test_index
        self.window_size = window_size
        self.max_seq_len = max_seq_len
        self.max_subseq_len = max_subseq_len
        self.metric = metric
        self.sse_prob = sse_prob
        self.sse_type = sse_type
        self.question_transition = question_transition
        self.lecture_transition = lecture_transition
        self.multitype_transition = multitype_transition
        self.user_id_mapping = {}
        all_user_records = {**train_user_records, **test_user_records}
        # all_user_records = {**test_user_records}
        outputs = self._transform(all_user_records)
        self.q_data, self.a_data, self.l_data = outputs[:3]
        self.train_mask_data, self.val_mask_data, self.test_mask_data = outputs[3:]

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
        lectures = self.l_data[idx]
        train_target_mask = self.train_mask_data[idx]
        val_target_mask = self.val_mask_data[idx]
        test_target_mask = self.test_mask_data[idx]
        assert len(questions) == len(answers) == len(lectures)

        user = self.user_id_mapping[idx]
        if self.metric == "rmse":
            interactions = []
            for i in range(len(questions)):
                q = questions[i]
                if q != 0:  # if q == 0, that is padding
                    if np.random.random() < self.sse_prob:
                        if self.sse_type == "random":
                            q = np.random.randint(1, self.num_items + 1)
                            questions[i] = q
                            a = np.random.choice([0, 1], p=[0.5, 0.5])
                            answers[i] = a
                        else:
                            next_items = self.question_transition[q]
                            if len(next_items) == 0:
                                continue
                            j = np.random.choice(range(len(next_items)))
                            q, a = next_items[j]
                            questions[i] = q
                            answers[i] = a
                    interactions.append([q, answers[i]])
                    for j in range(len(lectures)):
                        for m in range(self.max_subseq_len):
                            l = lectures[j][m]
                            if l != 0:
                                if np.random.random() < self.sse_prob:
                                    if self.sse_type == "random":
                                        item = np.random.randint(
                                            1, self.num_items + self.num_nonassessed_items + 1
                                        )
                                        lectures[j][m] = item
                                    else:
                                        next_items = self.multitype_transition[l]
                                        if len(next_items) == 0:
                                            continue
                                        k = np.random.choice(range(len(next_items)))
                                        item, a = next_items[k]
                                        lectures[j][m] = item
                else:
                    interactions.append([0, 0])
            interactions = np.array(interactions, dtype=float)
        else:
            interactions = np.zeros_like(questions, dtype=int)
            for i in range(len(questions)):
                q = questions[i]
                if q != 0:
                    if np.random.random() < self.sse_prob:
                        if self.sse_type == "random":
                            q = np.random.randint(1, self.num_items + 1)
                            questions[i] = q
                            a = np.random.choice([0, 1], p=[0.5, 0.5])
                            answers[i] = a
                        else:
                            next_items = self.question_transition[q]
                            if len(next_items) == 0:
                                continue
                            j = np.random.choice(range(len(next_items)))
                            q, a = next_items[j]
                            questions[i] = q
                            answers[i] = a
                    interactions[i] = q + answers[i] * self.num_items

                    for j in range(len(lectures)):
                        for m in range(self.max_subseq_len):
                            l = lectures[j][m]
                            if l != 0:
                                if np.random.random() < self.sse_prob:
                                    if self.sse_type == "random":
                                        item = np.random.randint(
                                            1, self.num_items + self.num_nonassessed_items + 1
                                        )
                                        lectures[j][m] = item
                                    else:
                                        next_items = self.multitype_transition[l]
                                        if len(next_items) == 0:
                                            continue
                                        k = np.random.choice(range(len(next_items)))
                                        item, a = next_items[k]
                                        lectures[j][m] = item
                else:
                    interactions[i] = 0

        return questions, answers, lectures, user, interactions, train_target_mask, \
               val_target_mask, test_target_mask

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
        l_data = []
        train_mask_data = []
        val_mask_data = []
        test_mask_data = []
        window_padding = Padding(self.window_size, side='left', fillvalue=0)
        mask_padding = Padding(self.window_size, side='left', fillvalue=False)
        lec_padding = Padding(self.window_size, side='left', fillvalue=[0] * self.max_subseq_len)
        lec_sub_padding = Padding(self.max_subseq_len, side='left', fillvalue=0)
        for user in sorted(list(all_user_records.keys())):
            q_len = len(all_user_records[user]['q'])
            q_list = all_user_records[user]['q'].copy()
            a_list = all_user_records[user]['a'].copy()
            l_list = all_user_records[user]['l'].copy()
            assert len(q_list) == len(a_list) == len(l_list)
            if q_len <= self.current_test_index:
                continue
                # questions = q_list[-self.window_size:]
                # answers = a_list[-self.window_size:]
                # lectures = l_list[-self.window_size:]
                # train_mask = list((np.array(questions) != 0))
                # val_mask = [False] * len(questions)
                # test_mask = [False] * len(questions)
                # for i in range(3, len(questions), 3):
                #     train_mask[i] = False
                #     val_mask[i] = True
            else:
                if self.current_test_index + 1 >= self.window_size:
                    questions = q_list[self.current_test_index + 1 - self.window_size:
                                       self.current_test_index + 1]
                    answers = a_list[self.current_test_index + 1 - self.window_size:
                                     self.current_test_index + 1]
                    lectures = l_list[self.current_test_index + 1 - self.window_size:
                                      self.current_test_index + 1]
                    lectures = [lec_sub_padding({"l": l[-self.max_subseq_len:]})["l"] for l in
                                lectures]

                    if user in self.train_user_records:
                        train_mask = list((np.array(questions) != 0))
                        val_mask = [False] * self.window_size
                        test_mask = [False] * self.window_size
                        for i in range(3, self.window_size, 3):
                            train_mask[i] = False
                            val_mask[i] = True
                    else:
                        train_mask = [True] * self.window_size
                        val_mask = [False] * self.window_size
                        for i in range(3, self.window_size - 1, 3):
                            train_mask[i] = False
                            val_mask[i] = True
                        train_mask[-1] = False
                        test_mask = [False] * self.window_size
                        test_mask[-1] = True
                else:
                    questions = q_list[:self.current_test_index + 1]
                    answers = a_list[:self.current_test_index + 1]
                    lectures = l_list[:self.current_test_index + 1]
                    # print(questions)
                    # print(answers)
                    if user in self.train_user_records:
                        train_mask = list((np.array(questions) != 0))
                        # print(train_mask)
                        val_mask = [False] * self.window_size
                        test_mask = [False] * self.window_size
                        for i in range(3, len(questions), 3):
                            train_mask[i] = False
                            val_mask[i] = True
                    else:
                        train_mask = [True] * len(questions)
                        val_mask = [False] * len(questions)
                        for i in range(3, len(questions) - 1, 3):
                            train_mask[i] = False
                            val_mask[i] = True
                        train_mask[-1] = False
                        test_mask = [False] * len(questions)
                        test_mask[-1] = True

            sample = {"q": questions, "a": answers}
            output = window_padding(sample)
            questions = output['q']
            answers = output['a']
            l_pad = [lec_sub_padding({"l": l[-self.max_subseq_len:]})["l"] for l in lectures]
            sample = {"l": l_pad}
            lec_output = lec_padding(sample)
            lectures = lec_output["l"]

            mask_sample = {"train": train_mask, "val": val_mask, "test": test_mask}
            mask_output = mask_padding(mask_sample)
            train_mask = mask_output["train"]
            val_mask = mask_output["val"]
            test_mask = mask_output["test"]

            id = len(q_data)
            self.user_id_mapping[id] = user
            q_data.append(questions)
            a_data.append(answers)
            l_data.append(lectures)
            train_mask_data.append(train_mask)
            val_mask_data.append(val_mask)
            test_mask_data.append(test_mask)

        return np.array(q_data), np.array(a_data), np.array(l_data), np.array(train_mask_data), \
               np.array(val_mask_data), np.array(test_mask_data)

    # def __getitem__(self, idx):
    #     """
    #     for online learning version
    #     idx: sample index
    #     """
    #     questions = self.q_data[idx]
    #     answers = self.a_data[idx]
    #     lectures = self.l_data[idx]
    #     assert len(questions) == len(answers) == len(lectures)
    #
    #     user = self.user_id_mapping[idx]
    #     if self.metric == "rmse":
    #         interactions = []
    #         for i in range(len(questions)):
    #             q = questions[i]
    #             if q != 0:  # if q == 0, that is padding
    #                 if np.random.random() < self.sse_prob:
    #                     if self.sse_type == "random":
    #                         q = np.random.randint(1, self.num_items + 1)
    #                         questions[i] = q
    #                         a = np.random.choice([0, 1], p=[0.5, 0.5])
    #                         answers[i] = a
    #                     else:
    #                         next_items = self.question_transition[q]
    #                         if len(next_items) == 0:
    #                             continue
    #                         j = np.random.choice(range(len(next_items)))
    #                         q, a = next_items[j]
    #                         questions[i] = q
    #                         answers[i] = a
    #                 interactions.append([q, answers[i]])
    #                 for j in range(len(lectures)):
    #                     for m in range(self.max_subseq_len):
    #                         l = lectures[j][m]
    #                         if l != 0:
    #                             if np.random.random() < self.sse_prob:
    #                                 if self.sse_type == "random":
    #                                     item = np.random.randint(
    #                                         1, self.num_items + self.num_nonassessed_items + 1
    #                                     )
    #                                     lectures[j][m] = item
    #                                 else:
    #                                     # next_items = self.multitype_transition[l]
    #                                     next_items = self.lecture_transition[l]
    #                                     if len(next_items) == 0:
    #                                         continue
    #                                     k = np.random.choice(range(len(next_items)))
    #                                     item, a = next_items[k]
    #                                     lectures[j][m] = item
    #             else:
    #                 interactions.append([0, 0])
    #         interactions = np.array(interactions, dtype=float)
    #     else:
    #         interactions = np.zeros_like(questions, dtype=int)
    #         for i in range(len(questions)):
    #             q = questions[i]
    #             if q != 0:
    #                 if np.random.random() < self.sse_prob:
    #                     if self.sse_type == "random":
    #                         q = np.random.randint(1, self.num_items + 1)
    #                         questions[i] = q
    #                         a = np.random.choice([0, 1], p=[0.5, 0.5])
    #                         answers[i] = a
    #                     else:
    #                         next_items = self.question_transition[q]
    #                         if len(next_items) == 0:
    #                             continue
    #                         j = np.random.choice(range(len(next_items)))
    #                         q, a = next_items[j]
    #                         questions[i] = q
    #                         answers[i] = a
    #                 interactions[i] = q + answers[i] * self.num_items
    #                 for j in range(len(lectures)):
    #                     for m in range(self.max_subseq_len):
    #                         l = lectures[j][m]
    #                         if l != 0:
    #                             if np.random.random() < self.sse_prob:
    #                                 if self.sse_type == "random":
    #                                     item = np.random.randint(
    #                                         1, self.num_items + self.num_nonassessed_items + 1
    #                                     )
    #                                     lectures[j][m] = item
    #                                 else:
    #                                     # next_items = self.multitype_transition[l]
    #                                     next_items = self.lecture_transition[l]
    #                                     if len(next_items) == 0:
    #                                         continue
    #                                     k = np.random.choice(range(len(next_items)))
    #                                     item, a = next_items[k]
    #                                     lectures[j][m] = item
    #             else:
    #                 interactions[i] = 0
    #
    #     questions = np.insert(questions, 0, 0)
    #     answers = np.insert(answers, 0, 0)
    #     if self.metric == "rmse":
    #         interactions = np.insert(interactions, 0, [0, 0], axis=0)
    #     else:
    #         interactions = np.insert(interactions, 0, 0)
    #     lectures = np.insert(lectures, 0, [0] * self.max_subseq_len, axis=0)
    #
    #     assert self.window_size <= len(questions)
    #
    #     if user in self.train_user_records:
    #         q_len = len(self.train_user_records[user]['q'])
    #         # note that len(questions) >= q_len + 1
    #         if q_len < self.window_size <= len(questions):
    #             questions = questions[:self.window_size]
    #             answers = answers[:self.window_size]
    #             interactions = interactions[:self.window_size]
    #             lectures = lectures[:self.window_size]
    #             target_mask = (questions != 0)
    #             train_target_mask = target_mask.copy()
    #             val_target_mask = np.array([False] * self.window_size)
    #             train_target_mask[q_len] = False
    #             val_target_mask[q_len] = True
    #             test_target_mask = np.array([False] * self.window_size)
    #         else:
    #             if self.current_test_index < (self.window_size // 2):
    #                 questions = questions[:self.window_size]
    #                 answers = answers[:self.window_size]
    #                 interactions = interactions[:self.window_size]
    #                 lectures = lectures[:self.window_size]
    #                 target_mask = (questions != 0)
    #                 train_target_mask = target_mask.copy()
    #                 val_target_mask = np.array([False] * self.window_size)
    #                 train_target_mask[-1] = False
    #                 val_target_mask[-1] = True
    #                 test_target_mask = np.array([False] * self.window_size)
    #             elif self.current_test_index > (q_len - self.window_size // 2):
    #                 # q_len is last index of questions
    #                 questions = questions[q_len - self.window_size + 1: q_len + 1]
    #                 answers = answers[q_len - self.window_size + 1: q_len + 1]
    #                 interactions = interactions[q_len - self.window_size + 1: q_len + 1]
    #                 lectures = lectures[q_len - self.window_size + 1: q_len + 1]
    #                 target_mask = (questions != 0)
    #                 train_target_mask = target_mask.copy()
    #                 val_target_mask = np.array([False] * self.window_size)
    #                 train_target_mask[-1] = False
    #                 val_target_mask[-1] = True
    #                 test_target_mask = np.array([False] * self.window_size)
    #             else:
    #                 start_index = self.current_test_index - (self.window_size // 2) + 1
    #                 questions = questions[start_index: start_index + self.window_size]
    #                 answers = answers[start_index: start_index + self.window_size]
    #                 interactions = interactions[start_index: start_index + self.window_size]
    #                 lectures = lectures[start_index: start_index + self.window_size]
    #                 target_mask = (questions != 0)
    #                 train_target_mask = target_mask.copy()
    #                 val_target_mask = np.array([False] * self.window_size)
    #                 train_target_mask[-1] = False
    #                 val_target_mask[-1] = True
    #                 test_target_mask = np.array([False] * self.window_size)
    #     elif user in self.test_user_records:
    #         q_len = len(self.test_user_records[user]['q'])
    #         if q_len < self.window_size <= len(questions):
    #             questions = questions[:self.window_size]
    #             answers = answers[:self.window_size]
    #             interactions = interactions[:self.window_size]
    #             lectures = lectures[:self.window_size]
    #             if self.current_test_index >= q_len:
    #                 # this test user's all data become training data
    #                 target_mask = (questions != 0)
    #                 train_target_mask = target_mask.copy()
    #                 val_target_mask = np.array([False] * self.window_size)
    #                 train_target_mask[q_len] = False
    #                 val_target_mask[q_len] = True
    #                 test_target_mask = np.array([False] * self.window_size)
    #             else:
    #                 train_target_mask = np.array([False] * self.window_size)
    #                 for i in range(1, self.current_test_index + 1, 1):
    #                     # because we insert 0 at the beginning, so iterate up to current_test_index
    #                     # questions[self.current_test_index + 1] is the test data point
    #                     train_target_mask[i] = True
    #                 val_target_mask = np.array([False] * self.window_size)
    #                 test_target_mask = np.array([False] * self.window_size)
    #                 test_target_mask[self.current_test_index + 1] = True
    #         else:
    #             # window_size <= q_len
    #             if self.current_test_index >= q_len:
    #                 # this test user's last window_size number of data become training data
    #                 questions = questions[q_len - self.window_size + 1: q_len + 1]
    #                 answers = answers[q_len - self.window_size + 1: q_len + 1]
    #                 interactions = interactions[q_len - self.window_size + 1: q_len + 1]
    #                 lectures = lectures[q_len - self.window_size + 1: q_len + 1]
    #                 target_mask = (questions != 0)
    #                 train_target_mask = target_mask.copy()
    #                 val_target_mask = np.array([False] * self.window_size)
    #                 train_target_mask[-1] = False
    #                 val_target_mask[-1] = True
    #                 test_target_mask = np.array([False] * self.window_size)
    #             else:
    #                 # when current_test_index < q_len
    #                 if self.current_test_index + 1 < self.window_size:
    #                     questions = questions[:self.window_size]
    #                     answers = answers[:self.window_size]
    #                     interactions = interactions[:self.window_size]
    #                     lectures = lectures[:self.window_size]
    #                     train_target_mask = np.array([False] * self.window_size)
    #                     for i in range(1, self.current_test_index + 1, 1):
    #                         train_target_mask[i] = True
    #                     val_target_mask = np.array([False] * self.window_size)
    #                     test_target_mask = np.array([False] * self.window_size)
    #                     test_target_mask[self.current_test_index + 1] = True
    #                 else:
    #                     start_index = self.current_test_index + 2 - self.window_size
    #                     questions = questions[start_index: self.current_test_index + 2]
    #                     answers = answers[start_index: self.current_test_index + 2]
    #                     interactions = interactions[start_index: self.current_test_index + 2]
    #                     lectures = lectures[start_index: self.current_test_index + 2]
    #                     train_target_mask = np.array([False] * self.window_size)
    #                     for i in range(0, self.window_size, 1):
    #                         train_target_mask[i] = True
    #                     val_target_mask = np.array([False] * self.window_size)
    #                     test_target_mask = np.array([False] * self.window_size)
    #                     test_target_mask[-1] = True
    #     else:
    #         raise ValueError
    #
    #     # if np.random.random() < self.sse_prob:
    #     #     user = np.random.choice(list(self.user_id_mapping.keys()))
    #     # print("idx: {}, questions {}".format(idx, questions))
    #     # print("answers {}".format(answers))
    #     # print("interactions {}".format(interactions))
    #     # print("lectures {}".format(lectures))
    #     # print("train_target_masks {}".format(train_target_mask))
    #     # print("val_target_masks {}".format(val_target_mask))
    #     # print("test_target_masks {}".format(test_target_mask))
    #     # return questions, answers, user, interactions, lectures, train_target_mask, \
    #     #        val_target_mask, test_target_mask
    #     return questions, answers, lectures, user, interactions, train_target_mask, \
    #            val_target_mask, test_target_mask
    #
    # def _transform(self, all_user_records):
    #     """
    #     transform the data into feasible input of model,
    #     truncate the seq. if it is too long and
    #     pad the seq. with 0s if it is too short
    #     """
    #     q_data = []
    #     a_data = []
    #     l_data = []
    #     padding = Padding(self.max_seq_len, side='right', fillvalue=0)
    #     lec_padding = Padding(self.max_seq_len, side='right', fillvalue=[0] * self.max_subseq_len)
    #     lec_sub_padding = Padding(self.max_subseq_len, side='left', fillvalue=0)
    #     for user in sorted(list(all_user_records.keys())):
    #         q_list = all_user_records[user]['q']
    #         a_list = all_user_records[user]['a']
    #         l_list = all_user_records[user]['l']
    #         assert len(q_list) == len(a_list) == len(l_list)
    #         sample = {"q": q_list, "a": a_list}
    #         output = padding(sample)  # output['q'] is 1d list
    #         id = len(q_data)
    #         self.user_id_mapping[id] = user
    #         q_data.append(output['q'])
    #         a_data.append(output['a'])
    #         l_list = [lec_sub_padding({"l": l[-self.max_subseq_len:]})["l"] for l in l_list]
    #         sample = {"l": l_list}
    #         lec_output = lec_padding(sample)
    #         l_data.append(lec_output["l"])
    #     users = list(sorted(all_user_records.keys()))
    #     assert users == [i for i in range(1, len(all_user_records) + 1)]
    #     return np.array(q_data), np.array(a_data), np.array(l_data)


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
    # test_index = 1
    # window_size = 5

    # test_index = 12
    # window_size = 5
    #
    # test_index = 1
    # window_size = 12
    #
    # test_index = 12
    # window_size = 12
    #
    # test_index = 1
    # window_size = 25

    # test_index = 12
    # window_size = 25

    # test_index = 13
    # window_size = 15

    # test_index = 7
    # window_size = 15

    test_index = 9
    window_size = 7

    test_users_data = LLPKTDataset(train_user_records, test_user_records, num_items,
                                   current_test_index=test_index,
                                   window_size=window_size,
                                   max_seq_len=20,
                                   question_transition=transition,
                                   metric="auc",
                                   sse_prob=0.,
                                   sse_type="random",
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
