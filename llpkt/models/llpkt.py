import torch
import torch.nn as nn
from llpkt.utils.misc import *


class LLPKT(nn.Module):
    """
    Extension of Memory-Augmented Neural Network (MANN)
    """

    def __init__(self, config):
        super().__init__()
        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda & config.cuda
        if self.cuda:
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cuda")
            # torch.cuda.set_device(config.gpu_device)
            prBlue("Random Number: {} ".format(torch.randint(0, 100, (3,))))
        else:
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cpu")
            prBlue("Random Number: {} ".format(torch.randint(0, 100, (3,))))
        self.metric = config.metric

        # initialize the parameters
        self.num_users = config.num_users
        self.max_q_length = config.max_q_length
        self.user_embed_dim = config.user_embed_dim
        self.num_questions = config.num_items
        self.num_concepts = config.num_concepts
        self.key_dim = config.hidden_dim
        self.value_dim = config.hidden_dim
        self.summary_dim = config.hidden_dim
        self.dropout_rate = config.dropout_rate

        # initialize the layers
        self.key_matrix_embed = nn.Embedding(num_embeddings=self.num_concepts + 1,
                                             embedding_dim=self.key_dim)
        nn.init.kaiming_normal_(self.key_matrix_embed.weight)
        self.init_value_matrix_embed = nn.Embedding(
            num_embeddings=self.num_users + 1, embedding_dim=self.num_concepts * self.value_dim,
            padding_idx=0)

        # used to store memory of lifelong learners
        # self.window_size = config.window_size
        # self.memory_slots = torch.zeros((self.num_users + 1, self.num_concepts * self.value_dim,
        #                                  self.window_size)).to(self.device)
        nn.init.kaiming_normal_(self.init_value_matrix_embed.weight)

        self.u_embed_matrix = nn.Embedding(num_embeddings=self.num_users + 1,
                                           embedding_dim=self.user_embed_dim,
                                           padding_idx=0)
        self.q_embed_matrix = nn.Embedding(num_embeddings=self.num_questions + 1,
                                           embedding_dim=self.key_dim,
                                           padding_idx=0)
        if self.metric == "rmse":
            self.qa_embed_matrix = nn.Linear(2, self.value_dim)
        else:
            self.qa_embed_matrix = nn.Embedding(num_embeddings=2 * self.num_questions + 1,
                                                embedding_dim=self.value_dim,
                                                padding_idx=0)
        # self.qa_embed_matrix = nn.Linear(2, self.value_dim)

        # self.erase_linear = nn.Linear(self.value_dim, self.value_dim)
        # self.add_linear = nn.Linear(self.value_dim, self.value_dim)
        # self.summary_fc = nn.Linear(2 * self.key_dim + 2 * self.value_dim, self.summary_dim)

        self.erase_linear = nn.Linear(self.value_dim + self.user_embed_dim, self.value_dim)
        self.add_linear = nn.Linear(self.value_dim + self.user_embed_dim, self.value_dim)
        self.summary_fc = nn.Linear(self.key_dim + self.value_dim + self.user_embed_dim,
                                    self.summary_dim)
        # self.linear_out = nn.Linear(self.summary_dim, 1)

        linear_dim = 2 * self.value_dim + self.user_embed_dim
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm1 = nn.LayerNorm(linear_dim)
        self.layer_norm2 = nn.LayerNorm(linear_dim)
        self.linear_out = nn.Linear(linear_dim, 1)
        self.linear1 = nn.Linear(in_features=linear_dim, out_features=linear_dim, bias=True)
        self.linear2 = nn.Linear(in_features=linear_dim, out_features=linear_dim, bias=True)

        # initialize the activate functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, q_data, qa_data, users, mode, update_memory=False):
        """
        get output of the model with size (batch_size, seq_len)
        :param q_data: (batch_size, seq_len)
        :param qa_data: (batch_size, seq_len)
        :param l_data: (batch_size, seq_len, 128)
        :param users
        :param offset is used only in online mode
        :return:
        """
        if self.metric == "rmse":
            qa_data = qa_data.float()
        batch_size, seq_len = q_data.size(0), q_data.size(1)

        if mode == "online":
            self.value_matrix = self.init_value_matrix_embed(users).reshape(
                batch_size, self.num_concepts, self.value_dim)
            # # TODO this is not correct.
            # self.value_matrix = self.memory_slots[users, :, 0].reshape(
            #     batch_size, self.num_concepts, self.value_dim)
        elif mode == "offline":
            self.value_matrix = torch.Tensor(self.num_concepts, self.value_dim).to(self.device)
            nn.init.kaiming_normal_(self.value_matrix)
            self.value_matrix = self.value_matrix.clone().repeat(batch_size, 1, 1)
        else:
            raise ValueError

        u_embed = self.u_embed_matrix(users)
        q_embed_data = self.q_embed_matrix(q_data)
        qa_embed_data = self.qa_embed_matrix(qa_data)
        # split the data seq into chunk and process.py each question sequentially
        sliced_q_embed_data = torch.chunk(q_embed_data, seq_len, dim=1)
        sliced_qa_embed_data = torch.chunk(qa_embed_data, seq_len, dim=1)

        batch_pred = []
        for i in range(seq_len):
            q = sliced_q_embed_data[i].squeeze(1)  # (batch_size, key_dim)
            qa = sliced_qa_embed_data[i].squeeze(1)  # (batch_size, key_dim)
            q_correlation_weight = self.compute_correlation_weight(q)
            q_read_content = self.read(q_correlation_weight)

            mastery_level = torch.cat([u_embed, q_read_content, q], dim=1)
            mastery_level = self.layer_norm1(mastery_level)
            summary_output = self.dropout(self.linear2(nn.ReLU()(self.linear1(mastery_level))))
            # summary_output = self.dropout(self.linear1(mastery_level))
            # mastery_level = torch.cat([q_read_content, q, l_read_content, ls], dim=1)
            # mastery_level = torch.cat([q_read_content, q, l_read_content], dim=1)
            # mastery_level = torch.cat([q_read_content, q], dim=1)
            # summary_output = self.tanh(self.summary_fc(mastery_level))
            batch_sliced_pred = self.sigmoid(self.linear_out(summary_output))
            batch_pred.append(batch_sliced_pred)
            self.value_matrix = self.write(q_correlation_weight, qa, u_embed)
            # update the memory slots
            if mode == "online" and update_memory and i == 0:
                # self.memory_slots[users, :, i] = self.value_matrix.clone().reshape(
                #     batch_size, self.num_concepts * self.value_dim)
                self.init_value_matrix_embed.weight[users, :] = self.value_matrix.clone().reshape(
                    batch_size, self.num_concepts * self.value_dim
                )
        batch_pred = torch.cat(batch_pred, dim=-1)
        return batch_pred

    def compute_correlation_weight(self, query_embedded):
        """
        use dot product to find the similarity between question embedding and each concept
        embedding stored as key_matrix
        where key-matrix could be understood as all concept embedding covered by the course.

        query_embeded : (batch_size, concept_embedding_dim)
        key_matrix : (num_concepts, concept_embedding_dim)
        output: is the correlation distribution between question and all concepts
        """

        concept_ids = torch.tensor([i for i in range(1, self.num_concepts + 1)]).to(self.device)
        similarity = query_embedded @ self.key_matrix_embed(concept_ids).t()
        # similarity = query_embedded @ self.key_matrix.t()
        correlation_weight = torch.softmax(similarity, dim=1)
        return correlation_weight

    def read(self, correlation_weight):
        """
        read function is to read a student's knowledge level on part of concepts covered by a
        target question.
        we could view value_matrix as the latent representation of a student's knowledge
        in terms of all possible concepts.

        value_matrix: (batch_size, num_concepts, concept_embedding_dim)
        correlation_weight: (batch_size, num_concepts)
        """
        batch_size = self.value_matrix.size(0)
        value_matrix_reshaped = self.value_matrix.reshape(
            batch_size * self.num_concepts, self.value_dim
        )
        correlation_weight_reshaped = correlation_weight.reshape(batch_size * self.num_concepts, 1)
        # a (10,3) * b (10,1) = c (10, 3)is every row vector of a multiplies the row scalar of b
        # the multiplication below is to scale the memory embedding by the correlation weight
        rc = value_matrix_reshaped * correlation_weight_reshaped
        read_content = rc.reshape(batch_size, self.num_concepts, self.value_dim)
        read_content = torch.sum(read_content, dim=1)  # sum over all concepts

        return read_content

    def write(self, correlation_weight, interaction_embedded, user_embedded):
        """
        write function is to update memory based on the interaction
        value_matrix: (batch_size, memory_size, memory_state_dim)
        correlation_weight: (batch_size, memory_size)
        qa_embedded: (batch_size, memory_state_dim)
        """
        batch_size = self.value_matrix.size(0)
        # user_info = torch.cat([interaction_embedded, read_content], dim=1)
        user_info = torch.cat([interaction_embedded, user_embedded], dim=1)
        # user_info = interaction_embedded
        erase_vector = self.erase_linear(user_info)  # (batch_size, memory_state_dim)
        erase_signal = self.sigmoid(erase_vector)

        add_vector = self.add_linear(user_info)  # (batch_size, memory_state_dim)
        add_signal = self.tanh(add_vector)

        erase_reshaped = erase_signal.reshape(batch_size, 1, self.value_dim)
        cw_reshaped = correlation_weight.reshape(batch_size, self.num_concepts, 1)
        # the multiplication is to generate weighted erase vector for each memory cell
        # therefore, the size is (batch_size, memory_size, memory_state_dim)
        erase_mul = erase_reshaped * cw_reshaped
        memory_after_erase = self.value_matrix * (1 - erase_mul)

        add_reshaped = add_signal.reshape(batch_size, 1, self.value_dim)
        # the multiplication is to generate weighted add vector for each memory cell
        # therefore, the size is (batch_size, memory_size, memory_state_dim)
        add_memory = add_reshaped * cw_reshaped

        updated_memory = memory_after_erase + add_memory
        return updated_memory


#
# class LLPKTMultiType(nn.Module):
#     """
#     Extension of Memory-Augmented Neural Network (MANN)
#     """
#
#     def __init__(self, config):
#         super().__init__()
#         self.is_cuda = torch.cuda.is_available()
#         self.cuda = self.is_cuda & config.cuda
#         if self.cuda:
#             torch.manual_seed(config.seed)
#             torch.cuda.manual_seed(config.seed)
#             self.device = torch.device("cuda")
#             # torch.cuda.set_device(config.gpu_device)
#             prBlue("Random Number: {} ".format(torch.randint(0, 100, (3,))))
#         else:
#             torch.manual_seed(config.seed)
#             torch.cuda.manual_seed(config.seed)
#             self.device = torch.device("cpu")
#             prBlue("Random Number: {} ".format(torch.randint(0, 100, (3,))))
#         self.metric = config.metric
#
#         # initialize the parameters
#         self.num_users = config.num_users
#         self.max_q_length = config.max_q_length
#         self.user_embed_dim = config.user_embed_dim
#         self.num_questions = config.num_items
#         self.num_lectures = config.num_nonassessed_items
#         self.num_concepts = config.num_concepts
#         self.key_dim = config.hidden_dim
#         self.value_dim = config.hidden_dim
#         self.summary_dim = config.hidden_dim
#
#         # initialize the layers
#         self.key_matrix_embed = nn.Embedding(num_embeddings=self.num_concepts + 1,
#                                              embedding_dim=self.key_dim)
#         nn.init.kaiming_normal_(self.key_matrix_embed.weight)
#         self.init_value_matrix_embed = nn.Embedding(
#             num_embeddings=self.num_users + 1, embedding_dim=self.num_concepts * self.value_dim,
#             padding_idx=0)
#
#         nn.init.kaiming_normal_(self.init_value_matrix_embed.weight)
#
#         self.u_embed_matrix = nn.Embedding(num_embeddings=self.num_users + 1,
#                                            embedding_dim=self.user_embed_dim,
#                                            padding_idx=0)
#         self.q_embed_matrix = nn.Embedding(num_embeddings=self.num_questions + self.num_lectures +1,
#                                            embedding_dim=self.key_dim,
#                                            padding_idx=0)
#         if self.metric == "rmse":
#             self.qa_embed_matrix = nn.Linear(2, self.value_dim)
#         else:
#             self.qa_embed_matrix = nn.Embedding(num_embeddings=2 * self.num_questions + 1,
#                                                 embedding_dim=self.value_dim,
#                                                 padding_idx=0)
#         self.erase_linear = nn.Linear(self.value_dim + self.user_embed_dim, self.value_dim)
#         self.add_linear = nn.Linear(self.value_dim + self.user_embed_dim, self.value_dim)
#         self.summary_fc = nn.Linear(2*self.key_dim + 2*self.value_dim + self.user_embed_dim,
#                                     self.summary_dim)
#         self.linear_out = nn.Linear(self.summary_dim, 1)
#         # initialize the activate functions
#         self.sigmoid = nn.Sigmoid()
#         self.tanh = nn.Tanh()
#         self.softmax = nn.Softmax()
#
#         self.dropout_rate = config.dropout_rate
#         linear_dim = 4 * self.value_dim + self.user_embed_dim
#         self.dropout = nn.Dropout(self.dropout_rate)
#         self.layer_norm1 = nn.LayerNorm(linear_dim)
#         self.layer_norm2 = nn.LayerNorm(linear_dim)
#         self.linear_out = nn.Linear(linear_dim, 1)
#         self.ffn = nn.ModuleList([
#             nn.Linear(in_features=linear_dim, out_features=linear_dim, bias=True)
#             for x in range(2)
#         ])
#
#     def forward(self, q_data, qa_data, l_data, users, mode, update_memory=False):
#         """
#         get output of the model with size (batch_size, seq_len)
#         :param q_data: (batch_size, seq_len)
#         :param qa_data: (batch_size, seq_len)
#         :param l_data: (batch_size, seq_len, 128)
#         :param users
#         :param offset is used only in online mode
#         :return:
#         """
#         if self.metric == "rmse":
#             qa_data = qa_data.float()
#         batch_size, seq_len, lec_len = l_data.size(0), l_data.size(1), l_data.size(2)
#
#         if mode == "online":
#             self.value_matrix = self.init_value_matrix_embed(users).reshape(
#                 batch_size, self.num_concepts, self.value_dim)
#         elif mode == "offline":
#             self.value_matrix = torch.Tensor(self.num_concepts, self.value_dim).to(self.device)
#             nn.init.kaiming_normal_(self.value_matrix)
#             self.value_matrix = self.value_matrix.clone().repeat(batch_size, 1, 1)
#         else:
#             raise ValueError
#
#         u_embed = self.u_embed_matrix(users)
#         q_embed_data = self.q_embed_matrix(q_data)
#         qa_embed_data = self.qa_embed_matrix(qa_data)
#         # split the data seq into chunk and process.py each question sequentially
#         sliced_q_embed_data = torch.chunk(q_embed_data, seq_len, dim=1)
#         sliced_qa_embed_data = torch.chunk(qa_embed_data, seq_len, dim=1)
#
#         l_read_content = torch.Tensor(batch_size, self.value_dim).to(self.device)
#         ls = torch.Tensor(batch_size, self.value_dim).to(self.device)
#         sliced_l_data = torch.chunk(l_data, seq_len, dim=1)  # seq_len * (batch_size, 1, lec_len)
#
#         batch_pred = []
#         for i in range(seq_len):
#             q = sliced_q_embed_data[i].squeeze(1)  # (batch_size, key_dim)
#             qa = sliced_qa_embed_data[i].squeeze(1)  # (batch_size, key_dim)
#             q_correlation_weight = self.compute_correlation_weight(q)
#             q_read_content = self.read(q_correlation_weight)
#
#             # should not change the order of code
#             nn.init.zeros_(l_read_content)
#             nn.init.zeros_(ls)
#             # (batch_size, 128, value_dim)
#             l_embed_data = self.q_embed_matrix(sliced_l_data[i].squeeze(1).long())
#             # print(l_embed_data.shape)
#             sliced_l_embed_data = torch.chunk(l_embed_data, lec_len, dim=1)
#             # 128 * (batch_size, 1, value_dim)
#             for j in range(lec_len):
#                 l = sliced_l_embed_data[j].squeeze(1)  # (batch_size, value_dim)
#                 l_correlation_weight = self.compute_correlation_weight(l)
#                 l_read_content += self.read(l_correlation_weight)
#                 self.value_matrix = self.write(l_correlation_weight, l, u_embed)
#                 ls += l
#
#             mastery_level = torch.cat([u_embed, q_read_content, q, l_read_content, l], dim=1)
#             # summary_output = self.tanh(self.summary_fc(mastery_level))
#             mastery_level = self.layer_norm1(mastery_level)
#             summary_output = self.dropout(self.ffn[1](nn.ReLU()(self.ffn[0](mastery_level))))
#             batch_sliced_pred = self.sigmoid(self.linear_out(summary_output))
#             batch_pred.append(batch_sliced_pred)
#             self.value_matrix = self.write(q_correlation_weight, qa, u_embed)
#             # update the memory slots
#             if mode == "online" and update_memory and i == 0:
#                 self.init_value_matrix_embed.weight[users, :] = self.value_matrix.clone().reshape(
#                     batch_size, self.num_concepts * self.value_dim
#                 )
#         batch_pred = torch.cat(batch_pred, dim=-1)
#         return batch_pred
#
#     def compute_correlation_weight(self, query_embedded):
#         """
#         use dot product to find the similarity between question embedding and each concept
#         embedding stored as key_matrix
#         where key-matrix could be understood as all concept embedding covered by the course.
#
#         query_embeded : (batch_size, concept_embedding_dim)
#         key_matrix : (num_concepts, concept_embedding_dim)
#         output: is the correlation distribution between question and all concepts
#         """
#
#         concept_ids = torch.tensor([i for i in range(1, self.num_concepts + 1)]).to(self.device)
#         similarity = query_embedded @ self.key_matrix_embed(concept_ids).t()
#         # similarity = query_embedded @ self.key_matrix.t()
#         correlation_weight = torch.softmax(similarity, dim=1)
#         return correlation_weight
#
#     def read(self, correlation_weight):
#         """
#         read function is to read a student's knowledge level on part of concepts covered by a
#         target question.
#         we could view value_matrix as the latent representation of a student's knowledge
#         in terms of all possible concepts.
#
#         value_matrix: (batch_size, num_concepts, concept_embedding_dim)
#         correlation_weight: (batch_size, num_concepts)
#         """
#         batch_size = self.value_matrix.size(0)
#         value_matrix_reshaped = self.value_matrix.reshape(
#             batch_size * self.num_concepts, self.value_dim
#         )
#         correlation_weight_reshaped = correlation_weight.reshape(batch_size * self.num_concepts, 1)
#         # a (10,3) * b (10,1) = c (10, 3)is every row vector of a multiplies the row scalar of b
#         # the multiplication below is to scale the memory embedding by the correlation weight
#         rc = value_matrix_reshaped * correlation_weight_reshaped
#         read_content = rc.reshape(batch_size, self.num_concepts, self.value_dim)
#         read_content = torch.sum(read_content, dim=1)  # sum over all concepts
#
#         return read_content
#
#     def write(self, correlation_weight, interaction_embedded, user_embedded):
#         """
#         write function is to update memory based on the interaction
#         value_matrix: (batch_size, memory_size, memory_state_dim)
#         correlation_weight: (batch_size, memory_size)
#         qa_embedded: (batch_size, memory_state_dim)
#         """
#         batch_size = self.value_matrix.size(0)
#         # user_info = torch.cat([interaction_embedded, read_content], dim=1)
#         user_info = torch.cat([interaction_embedded, user_embedded], dim=1)
#         # user_info = interaction_embedded
#         erase_vector = self.erase_linear(user_info)  # (batch_size, memory_state_dim)
#         erase_signal = self.sigmoid(erase_vector)
#
#         add_vector = self.add_linear(user_info)  # (batch_size, memory_state_dim)
#         add_signal = self.tanh(add_vector)
#
#         erase_reshaped = erase_signal.reshape(batch_size, 1, self.value_dim)
#         cw_reshaped = correlation_weight.reshape(batch_size, self.num_concepts, 1)
#         # the multiplication is to generate weighted erase vector for each memory cell
#         # therefore, the size is (batch_size, memory_size, memory_state_dim)
#         erase_mul = erase_reshaped * cw_reshaped
#         memory_after_erase = self.value_matrix * (1 - erase_mul)
#
#         add_reshaped = add_signal.reshape(batch_size, 1, self.value_dim)
#         # the multiplication is to generate weighted add vector for each memory cell
#         # therefore, the size is (batch_size, memory_size, memory_state_dim)
#         add_memory = add_reshaped * cw_reshaped
#
#         updated_memory = memory_after_erase + add_memory
#         return updated_memory

class LLPKTMultiType(nn.Module):
    """
    No Personalization
    """

    def __init__(self, config):
        super().__init__()
        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda & config.cuda
        if self.cuda:
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cuda")
            # torch.cuda.set_device(config.gpu_device)
            prBlue("Random Number: {} ".format(torch.randint(0, 100, (3,))))
        else:
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cpu")
            prBlue("Random Number: {} ".format(torch.randint(0, 100, (3,))))
        self.metric = config.metric

        # initialize the parameters
        self.num_users = config.num_users
        self.max_q_length = config.max_q_length
        self.user_embed_dim = config.user_embed_dim
        self.num_questions = config.num_items
        self.num_lectures = config.num_nonassessed_items
        self.num_concepts = config.num_concepts
        self.key_dim = config.hidden_dim
        self.value_dim = config.hidden_dim
        self.summary_dim = config.hidden_dim

        # initialize the layers
        self.key_matrix_embed = nn.Embedding(num_embeddings=self.num_concepts + 1,
                                             embedding_dim=self.key_dim)
        nn.init.kaiming_normal_(self.key_matrix_embed.weight)
        self.init_value_matrix_embed = nn.Embedding(
            num_embeddings=self.num_users + 1, embedding_dim=self.num_concepts * self.value_dim,
            padding_idx=0)

        nn.init.kaiming_normal_(self.init_value_matrix_embed.weight)

        self.u_embed_matrix = nn.Embedding(num_embeddings=self.num_users + 1,
                                           embedding_dim=self.user_embed_dim,
                                           padding_idx=0)
        self.q_embed_matrix = nn.Embedding(
            num_embeddings=self.num_questions + self.num_lectures + 1,
            embedding_dim=self.key_dim,
            padding_idx=0)
        if self.metric == "rmse":
            self.qa_embed_matrix = nn.Linear(2, self.value_dim)
        else:
            self.qa_embed_matrix = nn.Embedding(num_embeddings=2 * self.num_questions + 1,
                                                embedding_dim=self.value_dim,
                                                padding_idx=0)
        self.erase_linear = nn.Linear(self.value_dim, self.value_dim)
        self.add_linear = nn.Linear(self.value_dim, self.value_dim)
        self.summary_fc = nn.Linear(2 * self.key_dim + 2 * self.value_dim,
                                    self.summary_dim)
        self.linear_out = nn.Linear(self.summary_dim, 1)
        # initialize the activate functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        self.dropout_rate = config.dropout_rate
        linear_dim = 4 * self.value_dim
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm1 = nn.LayerNorm(linear_dim)
        self.layer_norm2 = nn.LayerNorm(linear_dim)
        self.linear_out = nn.Linear(linear_dim, 1)
        self.ffn = nn.ModuleList([
            nn.Linear(in_features=linear_dim, out_features=linear_dim, bias=True)
            for x in range(2)
        ])

    def forward(self, q_data, qa_data, l_data, users, mode, update_memory=False):
        """
        get output of the model with size (batch_size, seq_len)
        :param q_data: (batch_size, seq_len)
        :param qa_data: (batch_size, seq_len)
        :param l_data: (batch_size, seq_len, 128)
        :param users
        :param offset is used only in online mode
        :return:
        """
        if self.metric == "rmse":
            qa_data = qa_data.float()
        batch_size, seq_len, lec_len = l_data.size(0), l_data.size(1), l_data.size(2)

        if mode == "online":
            # self.value_matrix = self.init_value_matrix_embed(users).reshape(
            #     batch_size, self.num_concepts, self.value_dim)
            self.value_matrix = torch.Tensor(self.num_concepts, self.value_dim).to(self.device)
            nn.init.kaiming_normal_(self.value_matrix)
            self.value_matrix = self.value_matrix.clone().repeat(batch_size, 1, 1)
        elif mode == "offline":
            self.value_matrix = torch.Tensor(self.num_concepts, self.value_dim).to(self.device)
            nn.init.kaiming_normal_(self.value_matrix)
            self.value_matrix = self.value_matrix.clone().repeat(batch_size, 1, 1)
        else:
            raise ValueError

        u_embed = self.u_embed_matrix(users)
        q_embed_data = self.q_embed_matrix(q_data)
        qa_embed_data = self.qa_embed_matrix(qa_data)
        # split the data seq into chunk and process.py each question sequentially
        sliced_q_embed_data = torch.chunk(q_embed_data, seq_len, dim=1)
        sliced_qa_embed_data = torch.chunk(qa_embed_data, seq_len, dim=1)

        l_read_content = torch.Tensor(batch_size, self.value_dim).to(self.device)
        ls = torch.Tensor(batch_size, self.value_dim).to(self.device)
        sliced_l_data = torch.chunk(l_data, seq_len, dim=1)  # seq_len * (batch_size, 1, lec_len)

        batch_pred = []
        for i in range(seq_len):
            q = sliced_q_embed_data[i].squeeze(1)  # (batch_size, key_dim)
            qa = sliced_qa_embed_data[i].squeeze(1)  # (batch_size, key_dim)
            q_correlation_weight = self.compute_correlation_weight(q)
            q_read_content = self.read(q_correlation_weight)

            # should not change the order of code
            nn.init.zeros_(l_read_content)
            nn.init.zeros_(ls)
            # (batch_size, 128, value_dim)
            l_embed_data = self.q_embed_matrix(sliced_l_data[i].squeeze(1).long())
            # print(l_embed_data.shape)
            sliced_l_embed_data = torch.chunk(l_embed_data, lec_len, dim=1)
            # 128 * (batch_size, 1, value_dim)
            for j in range(lec_len):
                l = sliced_l_embed_data[j].squeeze(1)  # (batch_size, value_dim)
                l_correlation_weight = self.compute_correlation_weight(l)
                l_read_content += self.read(l_correlation_weight)
                self.value_matrix = self.write(l_correlation_weight, l, u_embed)
                ls += l

            # mastery_level = torch.cat([u_embed, q_read_content, q, l_read_content, l], dim=1)
            mastery_level = torch.cat([q_read_content, q, l_read_content, l], dim=1)
            # summary_output = self.tanh(self.summary_fc(mastery_level))
            mastery_level = self.layer_norm1(mastery_level)
            summary_output = self.dropout(self.ffn[1](nn.ReLU()(self.ffn[0](mastery_level))))
            batch_sliced_pred = self.sigmoid(self.linear_out(summary_output))
            batch_pred.append(batch_sliced_pred)
            self.value_matrix = self.write(q_correlation_weight, qa, u_embed)
            # update the memory slots
            # if mode == "online" and update_memory and i == 0:
            #     self.init_value_matrix_embed.weight[users, :] = self.value_matrix.clone().reshape(
            #         batch_size, self.num_concepts * self.value_dim
            #     )
        batch_pred = torch.cat(batch_pred, dim=-1)
        return batch_pred

    def compute_correlation_weight(self, query_embedded):
        """
        use dot product to find the similarity between question embedding and each concept
        embedding stored as key_matrix
        where key-matrix could be understood as all concept embedding covered by the course.

        query_embeded : (batch_size, concept_embedding_dim)
        key_matrix : (num_concepts, concept_embedding_dim)
        output: is the correlation distribution between question and all concepts
        """

        concept_ids = torch.tensor([i for i in range(1, self.num_concepts + 1)]).to(self.device)
        similarity = query_embedded @ self.key_matrix_embed(concept_ids).t()
        # similarity = query_embedded @ self.key_matrix.t()
        correlation_weight = torch.softmax(similarity, dim=1)
        return correlation_weight

    def read(self, correlation_weight):
        """
        read function is to read a student's knowledge level on part of concepts covered by a
        target question.
        we could view value_matrix as the latent representation of a student's knowledge
        in terms of all possible concepts.

        value_matrix: (batch_size, num_concepts, concept_embedding_dim)
        correlation_weight: (batch_size, num_concepts)
        """
        batch_size = self.value_matrix.size(0)
        value_matrix_reshaped = self.value_matrix.reshape(
            batch_size * self.num_concepts, self.value_dim
        )
        correlation_weight_reshaped = correlation_weight.reshape(batch_size * self.num_concepts, 1)
        # a (10,3) * b (10,1) = c (10, 3)is every row vector of a multiplies the row scalar of b
        # the multiplication below is to scale the memory embedding by the correlation weight
        rc = value_matrix_reshaped * correlation_weight_reshaped
        read_content = rc.reshape(batch_size, self.num_concepts, self.value_dim)
        read_content = torch.sum(read_content, dim=1)  # sum over all concepts

        return read_content

    def write(self, correlation_weight, interaction_embedded, user_embedded):
        """
        write function is to update memory based on the interaction
        value_matrix: (batch_size, memory_size, memory_state_dim)
        correlation_weight: (batch_size, memory_size)
        qa_embedded: (batch_size, memory_state_dim)
        """
        batch_size = self.value_matrix.size(0)
        # user_info = torch.cat([interaction_embedded, read_content], dim=1)
        # user_info = torch.cat([interaction_embedded, user_embedded], dim=1)
        user_info = interaction_embedded
        erase_vector = self.erase_linear(user_info)  # (batch_size, memory_state_dim)
        erase_signal = self.sigmoid(erase_vector)

        add_vector = self.add_linear(user_info)  # (batch_size, memory_state_dim)
        add_signal = self.tanh(add_vector)

        erase_reshaped = erase_signal.reshape(batch_size, 1, self.value_dim)
        cw_reshaped = correlation_weight.reshape(batch_size, self.num_concepts, 1)
        # the multiplication is to generate weighted erase vector for each memory cell
        # therefore, the size is (batch_size, memory_size, memory_state_dim)
        erase_mul = erase_reshaped * cw_reshaped
        memory_after_erase = self.value_matrix * (1 - erase_mul)

        add_reshaped = add_signal.reshape(batch_size, 1, self.value_dim)
        # the multiplication is to generate weighted add vector for each memory cell
        # therefore, the size is (batch_size, memory_size, memory_state_dim)
        add_memory = add_reshaped * cw_reshaped

        updated_memory = memory_after_erase + add_memory
        return updated_memory
