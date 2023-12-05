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

        # self.mu = nn.Linear(self.hidden_dim * 2, self.z_dim)
        # self.sigma = nn.Linear(self.hidden_dim * 2, self.z_dim)
        self.enc_out = nn.Linear(self.hidden_dim * 2, self.z_dim * 2)

        # nn.init.normal_(self.mu.weight, mean=0, std=0.001)
        # nn.init.normal_(self.sigma.weight, mean=0, std=0.001)

    def forward(self, x):
        x, _ = self.biLstm(x)  # [batch, seq_len, 2*enc_hidden_dim]
        # print(x.shape)
        x = self.enc_out(x)  # [batch, seq_len, 2*z_dim]
        mu, logvar = torch.chunk(x, 2, dim=-1)
        logvar = F.softplus(logvar)
        sigma = torch.exp(logvar * 2)

        # final_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # mu = self.mu(final_hidden)
        # sigma = F.softplus(self.sigma(final_hidden))
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
            h0 = torch.tanh(self.fc_latent(z)).unsqueeze(0).repeat(2, 1, 1)
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
                logits = self.output_layer(output.squeeze(1))

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

        self.config = config

        # Initial fully connected layer for the latent vector
        # TODO: this needs to change if I want to support multilevel conductor LSTMs,
        # since the hidden state size is multiplied by the number of layers and in the format
        # (layers, batch, hidden_dim)
        self.fc_z = nn.Linear(config.z_dim, config.conductor_hidden_dim)

        # Conductor RNN
        self.conductor_rnn = nn.LSTM(
            input_size=config.conductor_hidden_dim,
            hidden_size=config.conductor_hidden_dim,
            num_layers=config.conductor_layers,
            batch_first=True,
        )

        # Fully connected layer to transform conductor output to embedding_dim
        self.fc_conductor_output = nn.Linear(
            config.conductor_hidden_dim, config.embedding_dim
        )

        # Decoder RNN
        self.decoder_rnn = nn.LSTM(
            input_size=config.vocab_size + config.embedding_dim,
            hidden_size=config.decoder_hidden_dim,
            num_layers=config.decoder_layers,
            batch_first=True,
            dropout=0.5,
        )

        # Output layer
        self.fc_logits = nn.Linear(config.decoder_hidden_dim, config.vocab_size)

    def forward(self, z, x):
        conductor_input = self.fc_z(z)
        print(conductor_input.shape)
        # conductor_input: [batch_size, seq_len, latent_dim]
        batch_size = z.shape[0]

        # Prepare the latent vector as the initial state of the conductor
        conductor_h = torch.zeros(1, z.shape[0], self.config.conductor_hidden_dim).to(
            ut.get_device()
        )
        conductor_c = torch.zeros_like(conductor_h).to(ut.get_device())

        # Use the conductor to generate num_subsequences embedding vectors
        # conductor_input = torch.zeros(
        #     batch_size, 1, self.config.conductor_hidden_dim
        # ).to(ut.get_device())

        # print(conductor_input.size())
        # print(self.config.conductor_output_dim)

        conductor_outputs = []
        for i in range(self.config.num_subsequences):
            conductor_output, (conductor_h, conductor_c) = self.conductor_rnn(
                conductor_input[:, i, :].unsqueeze(1), (conductor_h, conductor_c)
            )
            conductor_outputs.append(conductor_output)
            # print(conductor_input.size())

        conductor_outputs = torch.stack(conductor_outputs, dim=1)
        print(conductor_outputs.shape)
        conductor_outputs = conductor_outputs.view(
            batch_size * self.config.num_subsequences, -1
        )
        embeddings = self.fc_conductor_output(conductor_outputs)
        embeddings = torch.tanh(embeddings)
        embeddings = embeddings.view(batch_size, self.config.num_subsequences, -1)

        # Decode each subsequence using embeddings
        logits = []
        prev_token = torch.zeros(batch_size, 1, self.config.vocab_size).to(
            ut.get_device()
        )
        hidden = (
            torch.zeros(
                self.config.decoder_layers, batch_size, self.config.decoder_hidden_dim
            ).to(ut.get_device()),
            torch.zeros(
                self.config.decoder_layers, batch_size, self.config.decoder_hidden_dim
            ).to(ut.get_device()),
        )
        for t in range(self.config.num_subsequences):
            curr_embedding = embeddings.select(1, t).unsqueeze(1)
            # print(prev_token.size())
            # print(curr_embedding.size())

            for n in range(self.config.subsequence_length):
                augmented_input = torch.cat((prev_token, curr_embedding), dim=-1)
                output, hidden = self.decoder_rnn(augmented_input, hidden)
                output = output.squeeze(1)
                logit = self.fc_logits(output).unsqueeze(1)
                logits.append(logit)

                if self.config.teacher_force:
                    curr_note_idx = t * self.config.num_subsequences + n
                    prev_token = x.select(1, curr_note_idx).unsqueeze(1)
                else:
                    prev_token = logit

        # Concatenate the outputs to form a continuous sequence
        # final_output shape: [batch_size, num_subsequences * subsequence_length, vocab_size]
        return torch.cat(logits, dim=1)

    def sample(self, z, start_token, temperature=1.0):
        generated_sequence = []

        # z: [batch_size, latent_dim]
        batch_size = z.shape[0]

        # Prepare the latent vector as the initial state of the conductor
        conductor_h = torch.tanh(self.fc_z(z)).unsqueeze(0)
        conductor_c = torch.zeros_like(conductor_h)

        # Use the conductor to generate num_subsequences embedding vectors
        conductor_input = torch.zeros(
            batch_size, 1, self.config.conductor_hidden_dim
        ).to(ut.get_device())

        # print(conductor_input.size())
        # print(self.config.conductor_output_dim)

        conductor_outputs = []
        for _ in range(self.config.num_subsequences):
            conductor_output, (conductor_h, conductor_c) = self.conductor_rnn(
                conductor_input, (conductor_h, conductor_c)
            )
            conductor_outputs.append(conductor_output)
            conductor_input = conductor_output
            # print(conductor_input.size())

        conductor_outputs = torch.stack(conductor_outputs, dim=1)
        conductor_outputs = conductor_outputs.view(
            batch_size * self.config.num_subsequences, -1
        )
        embeddings = self.fc_conductor_output(conductor_outputs)
        embeddings = torch.tanh(embeddings)
        embeddings = embeddings.view(batch_size, self.config.num_subsequences, -1)

        # Decode each subsequence using embeddings
        logits = []
        prev_token = torch.zeros(batch_size, 1, self.config.vocab_size).to(
            ut.get_device()
        )
        hidden = (
            torch.zeros(
                self.config.decoder_layers, batch_size, self.config.decoder_hidden_dim
            ).to(ut.get_device()),
            torch.zeros(
                self.config.decoder_layers, batch_size, self.config.decoder_hidden_dim
            ).to(ut.get_device()),
        )
        for t in range(self.config.num_subsequences):
            curr_embedding = embeddings.select(1, t).unsqueeze(1)
            # print(prev_token.size())
            # print(curr_embedding.size())

            for _ in range(self.config.subsequence_length):
                augmented_input = torch.cat((prev_token, curr_embedding), dim=-1)
                output, hidden = self.decoder_rnn(augmented_input, hidden)
                output = output.squeeze(1)
                logit = self.fc_logits(output)
                logit = logit / temperature

                print(logit.size())
                probabilities = F.softmax(logit, dim=-1)
                next_token = torch.multinomial(probabilities, 1)
                generated_sequence.append(next_token)
                prev_token = F.one_hot(
                    next_token, num_classes=self.config.vocab_size
                ).float()
                print(prev_token.size())

        # Concatenate the outputs to form a continuous sequence
        # final_output shape: [batch_size, num_subsequences * subsequence_length, vocab_size]
        return torch.cat(generated_sequence, dim=1)

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
