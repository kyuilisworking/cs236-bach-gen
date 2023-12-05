import numpy as np
import torch
import torch.nn.functional as F
from codebase import utils as ut
from torch import autograd, nn, optim
from torch.nn import functional as F


class LstmModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            batch_first=True,
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        logits = self.fc(out)
        return logits

    def sample(self, initial_note, sequence_length):
        """
        Generate a sequence of notes, starting from an initial note.

        :param initial_note: The starting note (one-hot encoded). Shape: [1, 1, input_dim]
        :param sequence_length: Total length of the sequence to generate.
        :return: Generated sequence of notes.
        """
        self.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # Disable gradient computation
            # Initialize the hidden and cell states
            h = torch.zeros(self.num_layers, 1, self.hidden_dim).to(initial_note.device)
            c = torch.zeros(self.num_layers, 1, self.hidden_dim).to(initial_note.device)

            with torch.no_grad():  # Disable gradient computation
                # Initialize the hidden and cell states
                h = torch.zeros(self.num_layers, 1, self.hidden_dim).to(
                    initial_note.device
                )
                c = torch.zeros(self.num_layers, 1, self.hidden_dim).to(
                    initial_note.device
                )

                # Initialize the generated sequence with the initial note
                generated_sequence = initial_note

                for _ in range(sequence_length - 1):
                    # Pass only the last note and the hidden states to the LSTM
                    output, (h, c) = self.lstm(generated_sequence[:, -1:, :], (h, c))
                    logits = self.fc(output)

                    # Convert logits to probabilities and sample a note
                    probabilities = torch.softmax(logits[:, -1, :], dim=-1)

                    # Set the highest probability to zero and re-normalize
                    probabilities[0, probabilities.argmax()] = 0
                    probabilities[0, probabilities.argmax()] = 0
                    probabilities /= probabilities.sum()

                    next_note = torch.multinomial(probabilities, num_samples=1)

                    # One-hot encode the sampled note
                    next_note_one_hot = torch.zeros_like(probabilities).scatter_(
                        -1, next_note, 1
                    )
                    # print(generated_sequence[:, -1:, :].shape)
                    # print(logits.shape)
                    # print(probabilities.shape)
                    # print(probabilities)
                    # print(next_note)
                    # print(next_note_one_hot.shape)

                    # Append the next note to the generated sequence
                    generated_sequence = torch.cat(
                        [generated_sequence, next_note_one_hot.unsqueeze(1)], dim=1
                    )

                return generated_sequence
