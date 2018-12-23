from typing import Sequence, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

torch.manual_seed(110)
# typing, everything in Python is Object.
tensor_activation = Callable[[torch.Tensor], torch.Tensor]


class LSTMClassifier(nn.Module):
    def __init__(self, device, num_words, embedding_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.device = device
        self.take_last = True
        self.embedding = nn.Embedding(num_words, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            batch_first=True, num_layers=1, bidirectional=False)
        self.fc = nn.Linear(hidden_size * 1, output_size)
        # init
        self.embedding.weight.data.requires_grad = True
        nn.init.xavier_normal_(self.embedding.weight)
        # orthogonal init for lstm cells, default xavier_normal is defined in source code
        nn.init.orthogonal_(self.lstm.all_weights[0][0])  # w_ih, (4 * hidden_size x input_size)
        nn.init.orthogonal_(self.lstm.all_weights[0][1])  # w_hh, (4 * hidden_size x hidden_size)
        nn.init.zeros_(self.lstm.all_weights[0][2])  # b_ih, (4 * hidden_size)
        nn.init.zeros_(self.lstm.all_weights[0][3])  # b_hh, (4 * hidden_size)

    def forward(self, sent, sent_length):
        sent_embed = self.embedding(sent)
        lstm_out = self.forward_lstm_with_var_length(sent_embed, sent_length)
        fc_out = self.fc(lstm_out)

        return F.log_softmax(fc_out, dim=-1)

    def forward_lstm_with_var_length(self, x, x_len):
        # 1. sort
        x_sort_idx = np.argsort(-x_len)
        x_unsort_idx = torch.LongTensor(np.argsort(x_sort_idx))
        x_len = x_len[x_sort_idx]
        x = x[torch.LongTensor(x_sort_idx)]
        # 2. pack
        x_emb_p = pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        # 3. forward lstm
        out_pack, (hn, cn) = self.lstm(x_emb_p)
        # 4. unsort h
        # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        hn = hn.permute(1, 0, 2)[x_unsort_idx]  # swap the first two dim
        hn = hn.permute(1, 0, 2)  # swap the first two again to recover

        if self.take_last:
            return hn.squeeze(0)
        else:
            # TODO test if ok
            # unpack: out
            out, _ = pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[x_unsort_idx]
            # unpack: c
            # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
            cn = cn.permute(1, 0, 2)[x_unsort_idx]  # swap the first two dim
            cn = cn.permute(1, 0, 2)  # swap the first two again to recover
            return out, (hn, cn)


class LSTMCellClassifier(nn.Module):
    def __init__(self, device, num_w_i, embedding_size=100, hidden_size=64):
        super(LSTMCellClassifier, self).__init__()
        self.device = device
        print(self.device)
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_w_i, embedding_size, padding_idx=0)
        self.embedding.weight.data.requires_grad = True
        self.lstm_cell = nn.LSTMCell(embedding_size, hidden_size)
        self.lstm_cell = self.lstm_cell.to(self.device)
        self.fc = nn.Linear(hidden_size * 1, 13 + 1)
        # init
        nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, sent):
        embed = self.embedding(sent)
        hi, ci = torch.zeros((embed.shape[0], self.hidden_size)), torch.zeros((embed.shape[0], self.hidden_size))
        hi = hi.to(self.device)
        ci = ci.to(self.device)
        lstm_out = []
        for i in range(11):
            hi, ci = self.lstm_cell(embed[:, i, :], (hi, ci))
            lstm_out.append(hi)

        fc_out = self.fc(lstm_out[-1])
        return F.log_softmax(fc_out, dim=-1)


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


if __name__ == '__main__':
    # unit test for FeedForward Class
    inputs = torch.randn((100, 256))
    net = FeedForward(input_dim=256,
                      num_layers=2,
                      hidden_dims=[128, 16],
                      activations=[nn.ReLU(), nn.ReLU()]
                      )

    print(net(inputs).shape)
    for name, param in net.named_parameters():
        print(name, param.shape)

