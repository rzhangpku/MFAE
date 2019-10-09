"""
Definition of custom layers for the ESIM model.
"""
# Aurelien Coet, 2018.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention
from .utils import sort_by_seq_lens, masked_softmax, weighted_sum, normal_softmax

# Class widely inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/modules/input_variational_dropout.py
class RNNDropout(nn.Dropout):
    """
    Dropout layer for the inputs of RNNs.

    Apply the same dropout mask to all the elements of the same sequence in
    a batch of sequences of size (batch, sequences_length, embedding_dim).
    """

    def forward(self, sequences_batch):
        """
        Apply dropout to the input batch of sequences.

        Args:
            sequences_batch: A batch of sequences of vectors that will serve
                as input to an RNN.
                Tensor of size (batch, sequences_length, emebdding_dim).

        Returns:
            A new tensor on which dropout has been applied.
        """
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0],
                                             sequences_batch.shape[-1])
        dropout_mask = nn.functional.dropout(ones, self.p, self.training,
                                             inplace=False)
        return dropout_mask.unsqueeze(1) * sequences_batch


# class TransformerEncoder(nn.Module):
#     """
#     RNN taking variable length padded sequences of vectors as input and
#     encoding them into padded sequences of vectors of the same length.
#
#     This module is useful to handle batches of padded sequences of vectors
#     that have different lengths and that need to be passed through a RNN.
#     The sequences are sorted in descending order of their lengths, packed,
#     passed through the RNN, and the resulting sequences are then padded and
#     permuted back to the original order of the input sequences.
#     """
#
#     def __init__(self,
#                  input_size,
#                  nhead=4,
#                  num_layers=1):
#
#         super(TransformerEncoder, self).__init__()
#
#         self.input_size = input_size
#         self.nhead = nhead
#         self.num_layers = num_layers
#         self._encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
#             self.input_size, nhead=nhead), num_layers=num_layers)
#
#     def forward(self, sequences_batch, sequences_lengths):
#         sequences_batch = sequences_batch.transpose(1, 0).contiguous()
#         outputs = self._encoder(sequences_batch)
#         outputs = outputs.transpose(1, 0).contiguous()
#
#         sorted_batch, sorted_lengths, _, restoration_idx =\
#             sort_by_seq_lens(outputs, sequences_lengths)
#         packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch, sorted_lengths, batch_first=True)
#         outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_batch, batch_first=True)
#         return outputs

class LinerEncoder(nn.Module):
    """
    RNN taking variable length padded sequences of vectors as input and
    encoding them into padded sequences of vectors of the same length.

    This module is useful to handle batches of padded sequences of vectors
    that have different lengths and that need to be passed through a RNN.
    The sequences are sorted in descending order of their lengths, packed,
    passed through the RNN, and the resulting sequences are then padded and
    permuted back to the original order of the input sequences.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 dropout=0.0):
        super(LinerEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self._encoder = nn.Linear(input_size, hidden_size*2)

    def forward(self, sequences_batch, sequences_lengths):
        """
        Args:
            sequences_batch: A batch of variable length sequences of vectors.
                The batch is assumed to be of size
                (batch, sequence, vector_dim).
            sequences_lengths: A 1D tensor containing the sizes of the
                sequences in the input batch.

        Returns:
            reordered_outputs: The outputs (hidden states) of the encoder for
                the sequences in the input batch, in the same order.
        """
        sorted_batch, sorted_lengths, _, restoration_idx =\
            sort_by_seq_lens(sequences_batch, sequences_lengths)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch,
                                                         sorted_lengths,
                                                         batch_first=True)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_batch,
                                                      batch_first=True)

        outputs = self._encoder(outputs)

        reordered_outputs = outputs.index_select(0, restoration_idx)

        return reordered_outputs

class LengthEncoder(nn.Module):

    def forward(self, sequences_batch, sequences_lengths):
        sorted_batch, sorted_lengths, _, restoration_idx =\
            sort_by_seq_lens(sequences_batch, sequences_lengths)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch,
                                                         sorted_lengths,
                                                         batch_first=True)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_batch,
                                                      batch_first=True)
        reordered_outputs = outputs.index_select(0, restoration_idx)

        return reordered_outputs

class Seq2SeqEncoder(nn.Module):
    """
    RNN taking variable length padded sequences of vectors as input and
    encoding them into padded sequences of vectors of the same length.

    This module is useful to handle batches of padded sequences of vectors
    that have different lengths and that need to be passed through a RNN.
    The sequences are sorted in descending order of their lengths, packed,
    passed through the RNN, and the resulting sequences are then padded and
    permuted back to the original order of the input sequences.
    """

    def __init__(self,
                 rnn_type,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bias=True,
                 dropout=0.0,
                 bidirectional=False):
        """
        Args:
            rnn_type: The type of RNN to use as encoder in the module.
                Must be a class inheriting from torch.nn.RNNBase
                (such as torch.nn.LSTM for example).
            input_size: The number of expected features in the input of the
                module.
            hidden_size: The number of features in the hidden state of the RNN
                used as encoder by the module.
            num_layers: The number of recurrent layers in the encoder of the
                module. Defaults to 1.
            bias: If False, the encoder does not use bias weights b_ih and
                b_hh. Defaults to True.
            dropout: If non-zero, introduces a dropout layer on the outputs
                of each layer of the encoder except the last one, with dropout
                probability equal to 'dropout'. Defaults to 0.0.
            bidirectional: If True, the encoder of the module is bidirectional.
                Defaults to False.
        """
        assert issubclass(rnn_type, nn.RNNBase),\
            "rnn_type must be a class inheriting from torch.nn.RNNBase"

        super(Seq2SeqEncoder, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional

        self._encoder = rnn_type(input_size,
                                 hidden_size,
                                 num_layers=num_layers,
                                 bias=bias,
                                 batch_first=True,
                                 dropout=dropout,
                                 bidirectional=bidirectional)

    def forward(self, sequences_batch, sequences_lengths):
        """
        Args:
            sequences_batch: A batch of variable length sequences of vectors.
                The batch is assumed to be of size
                (batch, sequence, vector_dim).
            sequences_lengths: A 1D tensor containing the sizes of the
                sequences in the input batch.

        Returns:
            reordered_outputs: The outputs (hidden states) of the encoder for
                the sequences in the input batch, in the same order.
        """
        sorted_batch, sorted_lengths, _, restoration_idx =\
            sort_by_seq_lens(sequences_batch, sequences_lengths)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch,
                                                         sorted_lengths,
                                                         batch_first=True)

        outputs, _ = self._encoder(packed_batch, None)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,
                                                      batch_first=True)
        reordered_outputs = outputs.index_select(0, restoration_idx)

        return reordered_outputs


class SoftmaxAttention(nn.Module):

    def __init__(self, hidden_size, dropout=0.5):
        super(SoftmaxAttention, self).__init__()
        # self.multi_head_attn = MultiheadAttention(hidden_size*2, 8)
        self.liner1 = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size // 2))
        self.liner2 = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size // 2))
        self.liner3 = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size // 2))
        self.liner4 = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size // 2))
        self.liner5 = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size // 2))
        self.liner6 = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size // 2))
        self.liner7 = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size // 2))
        self.liner8 = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size // 2))
        self.liner = nn.Sequential(nn.Linear(hidden_size * 4, hidden_size * 4), nn.ReLU(), RNNDropout(p=dropout)) # ,RNNDropout(p=dropout)

        # self._enhance = nn.Sequential(nn.Linear(2*7*2*hidden_size, 7*2*hidden_size), nn.ReLU(), RNNDropout(p=dropout))
        self._projection = nn.Sequential(nn.Linear(7*2*hidden_size, hidden_size), nn.ReLU(), RNNDropout(p=dropout))

        # self.Wb_inter = torch.nn.Parameter(torch.randn(hidden_size*2, hidden_size*2), requires_grad=True)
        # self.Wb_intra = torch.nn.Parameter(torch.randn(hidden_size * 2, hidden_size * 2), requires_grad=True)

    """
    Attention layer taking premises and hypotheses encoded by an RNN as input
    and computing the soft attention between their elements.
    The dot product of the encoded vectors in the premises and hypotheses is
    first computed. The softmax of the result is then used in a weighted sum
    of the vectors of the premises for each element of the hypotheses, and
    conversely for the elements of the premises.
    """

    def forward(self,
                premise_batch,
                premise_mask,
                hypothesis_batch,
                hypothesis_mask):
        """
        Args:
            premise_batch: A batch of sequences of vectors representing the
                premises in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            premise_mask: A mask for the sequences in the premise batch, to
                ignore padding data in the sequences during the computation of
                the attention.
            hypothesis_batch: A batch of sequences of vectors representing the
                hypotheses in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            hypothesis_mask: A mask for the sequences in the hypotheses batch,
                to ignore padding data in the sequences during the computation
                of the attention.

        Returns:
            attended_premises: The sequences of attention vectors for the
                premises in the input batch.
            attended_hypotheses: The sequences of attention vectors for the
                hypotheses in the input batch.
        """
        # dot attn
        enhanced_premises0, enhanced_hypotheses0 = self.dot_attn(premise_batch, premise_mask,
                hypothesis_batch, hypothesis_mask)
        # # bilinear attn
        # enhanced_premises1, enhanced_hypotheses1 = self.bilinear_attn(premise_batch, premise_mask,
        #         hypothesis_batch, hypothesis_mask)
        #
        # enhanced_premises = self._enhance(torch.cat((enhanced_premises0, enhanced_premises1), dim=-1))
        # enhanced_hypotheses = self._enhance(torch.cat((enhanced_hypotheses0, enhanced_hypotheses1), dim=-1))
        # print(enhanced_premises0.size())
        projected_premises = self._projection(enhanced_premises0)
        projected_hypotheses = self._projection(enhanced_hypotheses0)

        return projected_premises, projected_hypotheses

    # def bilinear_attn(self, premise_batch, premise_mask,
    #             hypothesis_batch, hypothesis_mask):
    #     # inter-attention Softmax attention weights.
    #     Wb_inter = self.Wb_inter.repeat(premise_batch.size()[0], 1, 1)
    #     Wb_intra = self.Wb_intra.repeat(premise_batch.size()[0], 1, 1)
    #
    #     similarity_matrix = premise_batch.bmm(Wb_inter).bmm(hypothesis_batch.transpose(2, 1).contiguous())
    #     prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
    #     hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)
    #     attended_premises = weighted_sum(hypothesis_batch, prem_hyp_attn, premise_mask)
    #     attended_hypotheses = weighted_sum(premise_batch, hyp_prem_attn, hypothesis_mask)
    #
    #     self_premises_matrix = premise_batch.bmm(Wb_intra).bmm(premise_batch.transpose(2, 1).contiguous())
    #     self_hypotheses_matrix = hypothesis_batch.bmm(Wb_intra).bmm(hypothesis_batch.transpose(2, 1).contiguous())
    #     self_premises_attn = normal_softmax(self_premises_matrix)
    #     self_hypotheses_attn = normal_softmax(self_hypotheses_matrix)
    #     self_premises = self_premises_attn.bmm(premise_batch)
    #     self_hypotheses = self_hypotheses_attn.bmm(hypothesis_batch)
    #
    #     # attn_importance
    #     premise_importance = torch.sum(self_premises_attn, dim=-2).unsqueeze(-1)
    #     hypotheses_importance = torch.sum(self_hypotheses_attn, dim=-2).unsqueeze(-1)
    #     inter_hypotheses_importance = torch.sum(prem_hyp_attn, dim=-2).unsqueeze(-1)
    #     inter_premise_importance = torch.sum(hyp_prem_attn, dim=-2).unsqueeze(-1)
    #
    #     enhanced_premises, enhanced_hypotheses = self.multi_importance(premise_importance, hypotheses_importance,
    #                                                         inter_premise_importance, inter_hypotheses_importance,
    #                                                         premise_batch, hypothesis_batch, attended_premises,
    #                                                         attended_hypotheses, self_premises, self_hypotheses)
    #
    #     return enhanced_premises, enhanced_hypotheses

    def dot_attn(self, premise_batch, premise_mask,
                hypothesis_batch, hypothesis_mask):
        sqrt_dim = np.sqrt(premise_batch.size()[2])
        # inter-attention Softmax attention weights.
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1).contiguous()) / sqrt_dim

        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)
        attended_premises = weighted_sum(hypothesis_batch, prem_hyp_attn, premise_mask)
        attended_hypotheses = weighted_sum(premise_batch, hyp_prem_attn, hypothesis_mask)

        self_premises_matrix = premise_batch.bmm(premise_batch.transpose(2, 1).contiguous()) / sqrt_dim
        self_hypotheses_matrix = hypothesis_batch.bmm(hypothesis_batch.transpose(2, 1).contiguous()) / sqrt_dim

        self_premises_attn = normal_softmax(self_premises_matrix)
        self_hypotheses_attn = normal_softmax(self_hypotheses_matrix)
        self_premises = self_premises_attn.bmm(premise_batch)
        self_hypotheses = self_hypotheses_attn.bmm(hypothesis_batch)

        # attn_importance max
        premise_importance = torch.sum(self_premises_attn, dim=-2).unsqueeze(-1)
        hypotheses_importance = torch.sum(self_hypotheses_attn, dim=-2).unsqueeze(-1)
        inter_hypotheses_importance = torch.sum(prem_hyp_attn, dim=-2).unsqueeze(-1)
        inter_premise_importance = torch.sum(hyp_prem_attn, dim=-2).unsqueeze(-1)

        enhanced_premises, enhanced_hypotheses = self.multi_importance(premise_importance, hypotheses_importance,
                                                            inter_premise_importance, inter_hypotheses_importance,
                                                            premise_batch, hypothesis_batch, attended_premises,
                                                            attended_hypotheses, self_premises, self_hypotheses)

        return enhanced_premises, enhanced_hypotheses


    def multi_importance(self, premise_importance, hypotheses_importance,
                         inter_premise_importance, inter_hypotheses_importance,
                         premise_batch, hypothesis_batch,attended_premises,
                         attended_hypotheses, self_premises, self_hypotheses):
        # attn1
        prem_all_attn1 = premise_importance * inter_premise_importance
        hyp_all_attn1 = hypotheses_importance * inter_hypotheses_importance
        attended_premises1 = self.liner1(premise_batch * prem_all_attn1)
        attended_hypotheses1 = self.liner1(hypothesis_batch * hyp_all_attn1)

        # attn2
        prem_all_attn2 = premise_importance + inter_premise_importance
        hyp_all_attn2 = hypotheses_importance + inter_hypotheses_importance
        attended_premises2 = self.liner2(premise_batch * prem_all_attn2)
        attended_hypotheses2 = self.liner2(hypothesis_batch * hyp_all_attn2)
        # attn3
        prem_all_attn3 = torch.max(premise_importance, inter_premise_importance)
        hyp_all_attn3 = torch.max(hypotheses_importance, inter_hypotheses_importance)
        attended_premises3 = self.liner3(premise_batch * prem_all_attn3)
        attended_hypotheses3 = self.liner3(hypothesis_batch * hyp_all_attn3)
        # attn4
        attended_premises4_1 = premise_batch * premise_importance
        attended_premises4_2 = premise_batch * inter_premise_importance
        attended_premises4 = self.liner4(torch.max(attended_premises4_1, attended_premises4_2))
        attended_hypotheses4_1 = hypothesis_batch * hypotheses_importance
        attended_hypotheses4_2 = hypothesis_batch * inter_hypotheses_importance
        attended_hypotheses4 = self.liner4(torch.max(attended_hypotheses4_1, attended_hypotheses4_2))
        # attn5
        attended_premises5 = self.liner5(premise_batch * (prem_all_attn1 + 1))
        attended_hypotheses5 = self.liner5(hypothesis_batch * (hyp_all_attn1 + 1))
        # attn6
        attended_premises6 = self.liner6(premise_batch * (prem_all_attn2 + 1))
        attended_hypotheses6 = self.liner6(hypothesis_batch * (hyp_all_attn2 + 1))
        # attn7
        attended_premises7 = self.liner7(premise_batch * (prem_all_attn3 + 1))
        attended_hypotheses7 = self.liner7(hypothesis_batch * (hyp_all_attn3 + 1))
        # attn8
        attended_premises8 = self.liner8(torch.max(attended_premises4_1, attended_premises4_2) + premise_batch)
        attended_hypotheses8 = self.liner8(torch.max(attended_hypotheses4_1, attended_hypotheses4_2) + hypothesis_batch)

        premise_all = self.liner(torch.cat([attended_premises1, attended_premises2, attended_premises3,
                                            attended_premises4, attended_premises5, attended_premises6,
                                            attended_premises7, attended_premises8], dim=-1))
        hypotheses_all = self.liner(torch.cat([attended_hypotheses1, attended_hypotheses2, attended_hypotheses3,
                                               attended_hypotheses4, attended_hypotheses5, attended_hypotheses6,
                                               attended_hypotheses7, attended_hypotheses8], dim=-1))

        enhanced_premises = torch.cat([premise_batch, attended_premises, self_premises, premise_all,
                                       premise_batch * attended_premises, premise_batch - attended_premises
                                       # self_premises - attended_premises, self_premises * attended_premises,
                                       ],
                                      dim=-1)

        enhanced_hypotheses = torch.cat([hypothesis_batch, attended_hypotheses, self_hypotheses, hypotheses_all,
                                         hypothesis_batch * attended_hypotheses, hypothesis_batch - attended_hypotheses
                                         # self_hypotheses - attended_hypotheses, self_hypotheses * attended_hypotheses,
                                         ],
                                        dim=-1)
        return enhanced_premises, enhanced_hypotheses