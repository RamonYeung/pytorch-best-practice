import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

torch.manual_seed(110)


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


if __name__ == '__main__':
    # unit test
    sentence = torch.tensor([[1, 2, 3],
                             [1, 2, 0]], dtype=torch.int64)
    m = LSTMClassifier('cpu', 5)

    print(m(sentence, np.array([3, 2])))
