import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F
from codebase.models.nns.params import EncoderConfig, DecoderConfig


class LstmBaseline(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, num_layers, nn="lstm", name="lstm_baseline"
    ):
        super().__init__()
        self.name = name
        nn = getattr(nns, nn)
        self.input_dim = input_dim
        self.lstm_model = nn.LstmModel(input_dim, hidden_dim, num_layers)

    def loss(self, x):
        # [batch_size, sequence_len, input_dim]
        logits = self.lstm_model(x)

        # [batch_size * sequence_len, input_dim]
        logits = logits.view(-1, logits.shape[-1])
        target_sequences = torch.argmax(x, dim=-1).view(-1)

        # calculate NLL
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, target_sequences)

        return loss

    def sample(self, initial_notes, sequence_len):
        """
        Generate a sequence of notes using the model.

        :param initial_notes: Tensor of initial notes to start generation. Shape: [1, initial_len, input_dim]
        :param sequence_length: Total length of the sequence to generate.
        :param input_dim: Dimension of each note (e.g., size of one-hot encoded vector).
        :return: Generated sequence of notes.
        """
        return self.lstm_model.sample(initial_notes, sequence_len)
        # self.lstm_model.eval()  # Set the model to evaluation mode

        # with torch.no_grad():
        #     # Initialize the sequence with the initial notes
        #     generated_sequence = initial_notes

        #     for _ in range(sequence_len - initial_notes.size(1)):
        #         # Predict the next note
        #         logits = self.lstm_model(generated_sequence)
        #         # print(logits)

        #         # Get the last note in the sequence
        #         last_logits = logits[:, -1, :]

        #         # Convert logits to probabilities and sample a note
        #         probabilities = torch.softmax(last_logits, dim=-1)
        #         next_note = torch.multinomial(probabilities, num_samples=1)

        #         # One-hot encode the sampled note to append it to the sequence
        #         next_note_one_hot = torch.zeros_like(probabilities).scatter_(
        #             -1, next_note, 1
        #         )

        #         # Append the next note to the sequence
        #         generated_sequence = torch.cat(
        #             [generated_sequence, next_note_one_hot.unsqueeze(1)], dim=1
        #         )

        #     return generated_sequence
