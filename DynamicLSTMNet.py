import torch
import torch.nn as nn

from easyfsl.utils import compute_backbone_output_shape


class DynamicLSTMNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(DynamicLSTMNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        # Enforce the same data type for LSTM weights
        self.lstm = self.lstm.float()

    def forward(self, x, seq_lengths):
        x = x.float()
        seq_lengths = seq_lengths.cpu()
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)

        # Get the max sequence length
        max_seq_length = seq_lengths.max().item()

        # Unpack the output
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=max_seq_length)

        avg_hidden_states = torch.mean(output, dim=1)
        max_hidden_states, _ = torch.max(output, dim=1)

        combined_hidden_states = torch.cat((avg_hidden_states, max_hidden_states), dim=1)

        return combined_hidden_states

    def compute_backbone_output_shape(self):
        return self.hidden_size
