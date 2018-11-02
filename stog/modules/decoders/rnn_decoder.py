import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNDecoderBase(torch.nn.Module):

    def __init__(self, rnn_cell, dropout):
        super(RNNDecoderBase, self).__init__()
        self.rnn_cell = rnn_cell
        self.dropout = dropout

    def forward(self, inputs, hidden_state):
        raise NotImplementedError


class InputFeedRNNDecoder(RNNDecoderBase):

    def __init__(self, rnn_cell, attention_layer, dropout):
        super(InputFeedRNNDecoder, self).__init__(rnn_cell, dropout)
        self.attention_layer = attention_layer

    def forward(self, inputs, memory_bank, mask, hidden_state):
        """

        :param inputs: [batch_size, decoder_seq_length, embedding_size]
        :param memory_bank: [batch_size, encoder_seq_length, encoder_hidden_size]
        :param mask:  None or [batch_size, decoder_seq_length]
        :param hidden_state: a tuple of (state, memory) with shape [num_encoder_layers, batch_size, encoder_hidden_size]
        :return:
        """
        batch_size = inputs.size(0)
        one_step_length = [1] * batch_size
        attentions = dict(
            std=[],
            copy=[]
        )
        output_sequences = []

        input_feed = inputs.new_zeros(batch_size, 1, self.module.hidden_size)

        for step_i, input in enumerate(inputs.split(1, dim=1)):
            # input: [batch_size, 1, embeddings_size]
            # input_feed: [batch_size, 1, hidden_size]
            _input = torch.cat([input, input_feed])
            packed_input = pack_padded_sequence(_input, one_step_length, batch_first=True)
            # hidden_state: a tuple of (state, memory) with shape [num_layers, batch_size, hidden_size]
            packed_output, hidden_state = self.rnn_cell(packed_input, hidden_state)
            # output: [batch_size, 1, hidden_size]
            output, _ = pad_packed_sequence(packed_output, batch_first=True)

            output, attention = self.attention_layer(output, memory_bank, mask)

            output = self.dropout(output)

            input_feed = output.clone()

            output_sequences.append(output)
            attentions['std'].append(attention)

        output_sequences = torch.cat(output_sequences, 1)
        return output_sequences, attentions, hidden_state