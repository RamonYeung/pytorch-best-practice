import math
from typing import Sequence, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

torch.manual_seed(10086)
# typing, everything in Python is Object.
tensor_activation = Callable[[torch.Tensor], torch.Tensor]


class FeedForward(nn.Module):
    """
    This part of code is taken from AllenNLP Package, and simplified a bit.
    Check it out.
    https://github.com/allenai/allennlp/blob/master/allennlp/modules/feedforward.py

    TODO: add dropout support.
    """

    def __init__(self,
                 input_dim: int,
                 num_layers: int,
                 hidden_dims: Union[int, Sequence[int]],
                 activations: Union[tensor_activation, Sequence[tensor_activation]]) -> None:

        super(FeedForward, self).__init__()

        # All the checking.
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * num_layers
        if not isinstance(activations, list):
            activations = [activations] * num_layers
        if len(hidden_dims) != num_layers:
            raise ValueError(f"len(hidden_dims) {len(hidden_dims)} != num_layers {num_layers}")
        if len(activations) != num_layers:
            raise ValueError(f"len(activations) {len(activations)} != num_layers {num_layers}")

        # init
        input_dims = [input_dim] + hidden_dims[:-1]
        linear_layers = []
        for layer_input_dim, layer_output_dim in zip(input_dims, hidden_dims):
            linear_layers.append(nn.Linear(layer_input_dim, layer_output_dim))

        self._linear_layers = nn.ModuleList(linear_layers)
        self._input_dim = input_dim
        self._output_dim = hidden_dims[-1]
        self._activations = activations

    def forward(self, inputs):
        output = inputs
        for layer, activation in zip(self._linear_layers, self._activations):
            output = activation(layer(output))
        return output

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim


class LSTM4VarLenSeq(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True, bidirectional=False, init='orthogonal', take_last=True):
        """
        no dropout support
        batch_first support deprecated, the input and output tensors are provided as (batch, seq_len, feature).

        :param init: ways to init the torch.nn.LSTM parameters, supports 'orthogonal' and 'uniform'
        :param take_last: 'True' if you only want the final hidden state otherwise 'False'
        """
        super(LSTM4VarLenSeq, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bias=bias,
                            bidirectional=bidirectional)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.bidirectional = bidirectional
        self.init = init
        self.take_last = take_last
        self.batch_first = True  # Please don't modify this

        self.init_parameters()

    def init_parameters(self):
        """orthogonal init yields generally good results than uniform init"""
        if self.init == 'orthogonal':
            gain = 1  # use default value
            for nth in range(self.num_layers * self.bidirectional):
                nn.init.orthogonal_(self.lstm.all_weights[nth][0], gain=gain)  # w_ih, (4 * hidden_size x input_size)
                nn.init.orthogonal_(self.lstm.all_weights[nth][1], gain=gain)  # w_hh, (4 * hidden_size x hidden_size)
                nn.init.zeros_(self.lstm.all_weights[nth][2])  # b_ih, (4 * hidden_size)
                nn.init.zeros_(self.lstm.all_weights[nth][3])  # b_hh, (4 * hidden_size)
        elif self.init == 'uniform':
            k = math.sqrt(1 / self.hidden_size)
            for nth in range(self.num_layers * self.bidirectional):
                nn.init.uniform_(self.lstm.all_weights[nth][0], -k, k)
                nn.init.uniform_(self.lstm.all_weights[nth][1], -k, k)
                nn.init.zeros_(self.lstm.all_weights[nth][2])
                nn.init.zeros_(self.lstm.all_weights[nth][3])
        else:
            raise NotImplemented('Unsupported Initialization')

    def forward(self, x, x_len, hx=None):
        # 1. Sort x and its corresponding length
        sorted_x_len, sorted_x_idx = torch.sort(x_len, descending=True)
        sorted_x = x[sorted_x_idx]
        # 2. Ready to unsort after LSTM forward pass
        # Note that PyTorch 0.4 has no argsort, but PyTorch 1.0 does.
        _, unsort_x_idx = torch.sort(sorted_x_idx, descending=False)

        # 3. Pack the sorted version of x and x_len, as required by the API.
        x_emb = pack_padded_sequence(sorted_x, sorted_x_len, batch_first=self.batch_first)

        # 4. Forward lstm
        # output_packed.data.shape is (seq_valid_data, num_directions * hidden_size).
        # See doc of torch.nn.LSTM for details.
        out_packed, (hn, cn) = self.lstm(x_emb)

        # 5. unsort h
        # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        hn = hn.permute(1, 0, 2)[unsort_x_idx]  # swap the first two dim
        hn = hn.permute(1, 0, 2)  # swap the first two again to recover
        if self.take_last:
            return hn.squeeze(0)
        else:
            # unpack: out
            # (batch, max_seq_len, num_directions * hidden_size)
            out, _ = pad_packed_sequence(out_packed, batch_first=self.batch_first)
            out = out[unsort_x_idx]
            # unpack: c
            # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
            cn = cn.permute(1, 0, 2)[unsort_x_idx]  # swap the first two dim
            cn = cn.permute(1, 0, 2)  # swap the first two again to recover
            return out, (hn, cn)


if __name__ == '__main__':
    # Note that in the future we will import unittest
    # and port the following examples to test folder.

    # Unit test for FeedForward Class
    # ================
    inputs = torch.randn((100, 256))
    net = FeedForward(input_dim=256,
                      num_layers=2,
                      hidden_dims=[128, 16],
                      activations=[nn.ReLU(), nn.ReLU()]
                      )

    print(net(inputs).shape)
    for name, param in net.named_parameters():
        print(name, param.shape)

    # Unit test for LSTM variable length sequences
    # ================
    net = LSTM4VarLenSeq(200, 100,
                         num_layers=3, bias=True, bidirectional=True, init='orthogonal', take_last=False)

    inputs = torch.tensor([[1, 2, 3, 0],
                           [2, 3, 0, 0],
                           [2, 4, 3, 0],
                           [1, 4, 3, 0],
                           [1, 2, 3, 4]])
    embedding = nn.Embedding(num_embeddings=5, embedding_dim=200, padding_idx=0)
    lens = torch.LongTensor([3, 2, 3, 3, 4])

    input_embed = embedding(inputs)
    output, (h, c) = net(input_embed, lens)
    print(output.shape)  # 5, 4, 200, batch, seq length, hidden_size * 2 (only last layer)
    print(h.shape)  # 6, 5, 100, num_layers * num_directions, batch, hidden_size
    print(c.shape)  # 6, 5, 100, num_layers * num_directions, batch, hidden_size
