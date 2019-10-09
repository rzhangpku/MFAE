"""
Definition of the ESIM model.
"""
# Aurelien Coet, 2018.

import torch
import torch.nn as nn
from .layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention, LengthEncoder
from .utils import get_mask, replace_masked
from allennlp.modules.elmo import Elmo, batch_to_ids

class ESIM(nn.Module):
    """
    Implementation of the ESIM model presented in the paper "Enhanced LSTM for
    Natural Language Inference" by Chen et al.
    """

    def __init__(self,
                 embedding_dim,
                 hidden_size,
                 options_file, weight_file,
                 padding_idx=0,
                 dropout=0.5,
                 num_classes=3,
                 device="cpu"):
        """
        Args:
            vocab_size: The size of the vocabulary of embeddings in the model.
            embedding_dim: The dimension of the word embeddings.
            hidden_size: The size of all the hidden layers in the network.
            embeddings: A tensor of size (vocab_size, embedding_dim) containing
                pretrained word embeddings. If None, word embeddings are
                initialised randomly. Defaults to None.
            padding_idx: The index of the padding token in the premises and
                hypotheses passed as input to the model. Defaults to 0.
            dropout: The dropout rate to use between the layers of the network.
                A dropout rate of 0 corresponds to using no dropout at all.
                Defaults to 0.5.
            num_classes: The number of classes in the output of the network.
                Defaults to 3.
            device: The name of the device on which the model is being
                executed. Defaults to 'cpu'.
        """
        super(ESIM, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device

        self.elmo = Elmo(options_file, weight_file, 2, requires_grad=False, dropout=self.dropout) #2

        # self.length_encoder = LengthEncoder()

        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)

        self._encoding = Seq2SeqEncoder(nn.LSTM,
                                        self.embedding_dim,
                                        self.hidden_size,
                                        bidirectional=True)

        self._attention = SoftmaxAttention(self.hidden_size)

        # self._projection = nn.Sequential(nn.Linear(7*2*self.hidden_size,
        #                                            self.hidden_size),
        #                                  nn.ReLU())

        self._composition = Seq2SeqEncoder(nn.LSTM,
                                           self.hidden_size,
                                           self.hidden_size,
                                           bidirectional=True)

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2*4*self.hidden_size,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       self.num_classes))


        # Initialize all weights and biases in the model.
        # self.apply(_init_esim_weights)

    def forward(self, premises_ids, hypotheses_ids):
        """
        Args:
            premises: A batch of varaible length sequences of word indices
                representing premises. The batch is assumed to be of size
                (batch, premises_length).
            premises_lengths: A 1D tensor containing the lengths of the
                premises in 'premises'.
            hypothesis: A batch of varaible length sequences of word indices
                representing hypotheses. The batch is assumed to be of size
                (batch, hypotheses_length).
            hypotheses_lengths: A 1D tensor containing the lengths of the
                hypotheses in 'hypotheses'.

        Returns:
            logits: A tensor of size (batch, num_classes) containing the
                logits for each output class of the model.
            probabilities: A tensor of size (batch, num_classes) containing
                the probabilities of each output class in the model.
        """
        premises = self.elmo(premises_ids)
        encoded_premises = premises['elmo_representations'][1]
        premises_mask = premises['mask'].float()
        premises_lengths = premises_mask.sum(1).long()
        # premises_mask = get_mask(premises_mask, premises_lengths).to(self.device)

        hypotheses = self.elmo(hypotheses_ids)
        encoded_hypotheses = hypotheses['elmo_representations'][1]
        hypotheses_mask = hypotheses['mask'].float()
        hypotheses_lengths = hypotheses_mask.sum(1).long()
        # hypotheses_mask = get_mask(hypotheses_mask, hypotheses_lengths).to(self.device)

        # encoded_premises = self.length_encoder(encoded_premises, premises_lengths)
        # encoded_hypotheses = self.length_encoder(encoded_hypotheses, hypotheses_lengths)
        encoded_premises = self._encoding(encoded_premises, premises_lengths)
        encoded_hypotheses = self._encoding(encoded_hypotheses, hypotheses_lengths)

        # enhanced_premises, enhanced_hypotheses = self._attention(encoded_premises, premises_mask,
        #                                                          encoded_hypotheses, hypotheses_mask)
        # projected_premises = self._projection(enhanced_premises)
        # projected_hypotheses = self._projection(enhanced_hypotheses)
        # if self.dropout:
        #     projected_premises = self._rnn_dropout(projected_premises)
        #     projected_hypotheses = self._rnn_dropout(projected_hypotheses)
        projected_premises, projected_hypotheses = self._attention(encoded_premises, premises_mask,
                                                                   encoded_hypotheses, hypotheses_mask)

        v_ai = self._composition(projected_premises, premises_lengths)
        v_bj = self._composition(projected_hypotheses, hypotheses_lengths)

        v_a_avg = torch.sum(v_ai * premises_mask.unsqueeze(1).transpose(2, 1), dim=1)\
            / torch.sum(premises_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * hypotheses_mask.unsqueeze(1).transpose(2, 1), dim=1)\
            / torch.sum(hypotheses_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        logits = self._classification(v)
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits, probabilities


def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0
