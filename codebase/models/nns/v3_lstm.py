# Copyright (c) 2021 Rui Shu
import numpy as np
import torch
import torch.nn.functional as F
from codebase import utils as ut
from torch import autograd, nn, optim
from torch.nn import functional as F
from codebase.models.nns.params import EncoderConfig, DecoderConfig


class BidirectionalLstmEncoder(nn.Module):
    def __init__(self, inputConfig: EncoderConfig):
        super().__init__()
        self.input_dim = inputConfig.input_dim
        self.hidden_dim = inputConfig.hidden_dim
        self.z_dim = inputConfig.z_dim

        self.biLstm = nn.LSTM(
            batch_first=True,
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            bidirectional=True,
        )

        self.mu = nn.Linear(self.hidden_dim * 2, self.z_dim)
        self.sigma = nn.Linear(self.hidden_dim * 2, self.z_dim)

        nn.init.normal_(self.mu.weight, mean=0, std=0.001)
        nn.init.normal_(self.sigma.weight, mean=0, std=0.001)

    def forward(self, x):
        _, (hidden, _) = self.biLstm(x)
        final_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        mu = self.mu(final_hidden)
        sigma = F.softplus(self.sigma(final_hidden))
        return mu, sigma


class CategoricalLstmDecoder(nn.Module):
    """
    1) given Z: pass into fully connected layer and tanh to get initial
    state for RNN
    2) iteratively sample from RNN to get a sequence of logits
    """

    def __init__(self, config: DecoderConfig):
        super(CategoricalLstmDecoder, self).__init__()

        self.config = config
        self.fc_latent = nn.Linear(config.z_dim, config.decoder_hidden_dim)

        # Decoder RNN
        self.decoder_rnn = nn.LSTM(
            input_size=config.vocab_size,
            hidden_size=config.decoder_hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        self.output_layer = nn.Linear(config.decoder_hidden_dim, config.vocab_size)

    def forward(self, z):
        # z: [batch_size, z_dim]
        batch_size = z.size(0)

        # Initialize the hidden state and cell state for LSTM
        # Process z through the fully connected layer and tanh activation
        h0 = torch.tanh(self.fc_latent(z)).unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.zeros_like(h0)

        # Prepare the initial input token
        input_token = (
            torch.zeros(batch_size, self.decoder_rnn.input_size)
            .unsqueeze(1)
            .to(z.device)
        )
        # Collect the output logits
        outputs = []

        sequence_length = self.config.subsequence_length * self.config.num_subsequences

        for _ in range(sequence_length):
            # h_n, c_n: [num_layers, batch_size, decoder_hidden_dim]
            output, (h0, c0) = self.decoder_rnn(input_token, (h0, c0))

            # Compute logits: [batch_size, vocab_size]
            logits = self.output_layer(output)

            # Append logits
            outputs.append(logits)

            # Update input token based on logits (softly)
            input_token = F.softmax(logits, dim=-1)

        # Stack outputs along sequence length
        outputs = torch.stack(outputs, dim=1)

        return outputs

    def sample(self, z, temperature=1.0):
        self.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # Disable gradient computation
            batch_size = z.size(0)
            h0 = (
                torch.tanh(self.fc_latent(z))
                .unsqueeze(0)
                .repeat(2, 1, self.decoder_rnn.hidden_size)
            )
            c0 = torch.zeros_like(h0)

            # Start with an initial input (can be a zero vector or any specific starting input)
            current_input = torch.zeros(batch_size, self.decoder_rnn.input_size).to(
                z.device
            )

            generated_sequence = []

            sequence_length = (
                self.config.subsequence_length * self.config.num_subsequences
            )

            for _ in range(sequence_length):
                output, (h0, c0) = self.decoder_rnn(
                    current_input.unsqueeze(1), (h0, c0)
                )
                logits = self.fc_out(output.squeeze(1))

                # Apply temperature scaling
                scaled_logits = logits / temperature

                # Convert logits to probabilities
                probabilities = F.softmax(scaled_logits, dim=-1)

                # Sample from the probability distribution
                next_token = torch.multinomial(probabilities, num_samples=1)

                generated_sequence.append(next_token)

                # Update the input for next time step
                current_input = (
                    F.one_hot(next_token, num_classes=self.decoder_rnn.input_size)
                    .float()
                    .squeeze(1)
                )

            # Concatenate the sequence along the sequence length dimension
            generated_sequence = torch.cat(generated_sequence, dim=1)

            return generated_sequence


# TODO: FIX
class HierarchicalLstmDecoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super(HierarchicalLstmDecoder, self).__init__()

        self.num_subsequences = config.num_subsequences
        self.subsequence_length = config.subsequence_length

        # Initial fully connected layer for the latent vector
        self.fc_latent = nn.Linear(config.z_dim, config.conductor_hidden_dim)

        # Conductor RNN
        self.conductor_rnn = nn.LSTM(
            input_size=config.conductor_hidden_dim,
            hidden_size=config.conductor_hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        # Fully connected layer to transform conductor output to decoder initial state
        self.fc_conductor_output = nn.Linear(
            config.conductor_hidden_dim, config.conductor_output_dim
        )

        # Decoder RNN
        self.decoder_rnn = nn.LSTM(
            input_size=config.vocab_size + config.conductor_output_dim,
            hidden_size=config.decoder_hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        # Output layer
        self.output_layer = nn.Linear(config.decoder_hidden_dim, config.vocab_size)

    def forward(self, z):
        # z: [batch_size, latent_dim]

        # Prepare the latent vector
        conductor_init_state = torch.tanh(self.fc_latent(z))
        conductor_init_state = (
            conductor_init_state.unsqueeze(0).repeat(2, 1, 1),
            conductor_init_state.unsqueeze(0).repeat(2, 1, 1),
        )

        # Decode each subsequence
        final_output = []
        for t in range(self.num_subsequences):
            embedding, conductor_state = self.conductor_rnn()
            # Prepare decoder input
            decoder_input = torch.cat(
                [
                    subsequence,
                    decoder_init_states[:, t, :]
                    .unsqueeze(1)
                    .repeat(1, subsequence.size(1), 1),
                ],
                dim=-1,
            )

            # Generate output for each subsequence
            output, _ = self.decoder_rnn(decoder_input)
            output = self.output_layer(output)
            final_output.append(output)

        # Concatenate the outputs to form a continuous sequence
        # final_output shape: [batch_size, num_subsequences * subsequence_length, vocab_size]
        return torch.cat(final_output, dim=1)

    def sample_x_given(self, z, start_token, temperature=1.0):
        """
        Generate a sequence given a latent vector z.

        :param z: Latent vector, shape [batch_size, latent_dim]
        :param start_token: Start token to begin generation
        :param temperature: Sampling temperature, higher values increase randomness
        :return: Generated sequence, shape [batch_size, num_subsequences * subsequence_length, vocab_size]
        """
        batch_size = z.size(0)
        generated_sequence = []

        # Prepare initial conductor state
        conductor_init_state = torch.tanh(self.fc_latent(z))
        conductor_init_state = (
            conductor_init_state.unsqueeze(0).repeat(2, 1, 1),
            conductor_init_state.unsqueeze(0).repeat(2, 1, 1),
        )

        # Generate conductor embeddings
        conductor_output, _ = self.conductor_rnn(
            self._init_sequence(batch_size, start_token), conductor_init_state
        )
        decoder_init_states = torch.tanh(self.fc_conductor_output(conductor_output))

        # Decode each subsequence
        for t in range(self.num_subsequences):
            # Prepare decoder input for the first time step
            previous_output = self._init_sequence(batch_size, start_token)
            decoder_input = torch.cat(
                [previous_output, decoder_init_states[:, t, :].unsqueeze(1)], dim=-1
            )

            # Generate subsequence
            subsequence = []
            for _ in range(self.subsequence_length):
                output, decoder_state = self.decoder_rnn(decoder_input)
                output = self.output_layer(output)

                # Apply temperature-based sampling
                output = output / temperature
                probabilities = F.softmax(output, dim=-1)
                next_token = torch.multinomial(probabilities.squeeze(1), 1)
                subsequence.append(next_token)

                # Prepare next input
                decoder_input = torch.cat(
                    [next_token, decoder_init_states[:, t, :]], dim=-1
                ).unsqueeze(1)

            # Concatenate generated tokens to form the subsequence
            subsequence = torch.cat(subsequence, dim=1)
            generated_sequence.append(subsequence)

        # Concatenate all subsequences to form the full sequence
        return torch.cat(generated_sequence, dim=1)

    def _init_sequence(self, batch_size, start_token):
        """
        Create an initial sequence tensor with the start token.

        :param batch_size: Batch size
        :param start_token: Start token to begin generation
        :return: Tensor with start tokens, shape [batch_size, 1, vocab_size]
        """
        start_tokens = torch.full(
            (batch_size, 1), start_token, dtype=torch.long, device=z.device
        )
        return F.one_hot(start_tokens, num_classes=self.vocab_size).float()
