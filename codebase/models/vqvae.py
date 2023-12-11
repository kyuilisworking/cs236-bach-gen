# Copyright (c) 2021 Rui Shu

import torch
from codebase import utils as ut
from codebase.models import nns
from torch.nn import functional as F
from codebase.models.nns.params import VQVAEConfig


class VQVAE(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_embeddings,
        commitment_cost,
        decay,
        nn="v4_vqvae",
        name="vqvae",
    ):
        super(VQVAE, self).__init__()
        self.name = name + "-" + str(num_embeddings)
        nn = getattr(nns, nn)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.quantizer = nn.VectorQuantizerEMA(
            num_embeddings, embedding_dim, commitment_cost, decay
        )
        # self.encoder = nn.EnhancedEncoder(embedding_dim)
        # self.decoder = nn.EnhancedDecoder(embedding_dim)
        self.encoder = nn.Encoder1D(embedding_dim)
        self.decoder = nn.Decoder1D(embedding_dim)
        # Reconstruction loss function
        self.reconstruction_loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        # Encoding
        encoded = self.encoder(x)

        # Quantization
        vq_loss, quantized, _ = self.quantizer(encoded)

        # Decoding
        decoded = self.decoder(quantized)

        return decoded, vq_loss

    # x: [batch, 1, 4, 88]
    def loss(self, x):
        decoded, vq_loss = self.forward(x)

        # decoded: [batch, 4, 88]
        decoded = decoded  # .unsqueeze(1)

        # Calculate reconstruction loss
        reconstruction_loss = self.reconstruction_loss_fn(decoded, x)

        # Total loss
        total_loss = vq_loss + reconstruction_loss

        return total_loss, vq_loss, reconstruction_loss

    def generate_embeddings_one_hot(self, x):
        """
        Convert input slice of four notes (multihot) to one-hot encodings based on the nearest embeddings in the codebook.
        :param inputs: A tensor of embeddings with shape [batch_size, 4, 88].
        :return: A tensor of one-hot encoded vectors with shape [batch_size, num_embeddings].
        """
        # Encoding
        encoded = self.encoder(x)

        # Quantization
        _, quantized, _ = self.quantizer(encoded)
        return self.encode_embedding_onehot(quantized)

    def encode_embedding_onehot(self, embeddings):
        """
        Convert input embeddings to one-hot encodings based on the nearest embeddings in the codebook.
        :param inputs: A tensor of embeddings with shape [batch_size, embedding_dim].
        :return: A tensor of one-hot encoded vectors with shape [batch_size, num_embeddings].
        """
        return self.quantizer.encode_one_hot(embeddings)

    def random_sample_one_hot_embedding(self, batch_size):
        """
        Generate a batch of random one-hot encoded vectors.

        :param batch_size: The number of one-hot encoded vectors in the batch.
        :return: A batch of one-hot encoded vectors, each with one random element set to 1.
        """
        # Create a batch of tensors of zeros
        one_hot_vectors = torch.zeros(batch_size, self.num_embeddings).to(
            ut.get_device()
        )

        # Randomly choose an index for each batch item to set to 1
        random_indices = torch.randint(0, self.num_embeddings, (batch_size,))
        one_hot_vectors[torch.arange(batch_size), random_indices] = 1

        return one_hot_vectors

    def embedding_to_notes(self, one_hot_embeddings):
        """
        Convert one-hot embeddings to notes by passing them through the decoder

        :param embeddings: [batch, codebook_dim]
        :return: [batch, 4, 88]
        """

        # First, convert all embeddings (one-hot) to actual embedding vectors
        embeddings = self.quantizer.decode_one_hot(
            one_hot_embeddings
        )  # [batch, embedding_dim]

        # Pass [batch, embedding_dim] through the decoder
        decoded = self.decoder(embeddings)

        probs = torch.sigmoid(decoded)

        top2 = torch.topk(probs, 2, dim=2)
        top2_indices = top2.indices
        top2_values = top2.values
        print(top2_values)
        print(top2_indices)

        # Create a tensor of zeros with the same shape as 'probs'
        transformed = torch.zeros_like(probs)

        # Define the threshold for "roughly the same"
        threshold_ratio = (
            0.95  # e.g., second probability should be at least 90% of the first
        )

        # Apply the logic
        for i in range(transformed.size(0)):  # Loop over the batch
            for j in range(transformed.size(1)):  # Loop over the second dimension
                # Always set the highest probability index to 1
                transformed[i, j, top2_indices[i, j, 0]] = 1

                # Set the second highest index to 1 only if it's within the threshold
                if top2_values[i, j, 1] >= threshold_ratio * top2_values[i, j, 0]:
                    transformed[i, j, top2_indices[i, j, 1]] = 1

        # Get the indices of the top 2 probabilities in each vector
        # top2_indices = torch.topk(probs, 2, dim=2).indices

        # # Create a tensor of zeros with the same shape as 'probabilities'
        # transformed = torch.zeros_like(probs)

        # # Set the elements at the top 2 indices to 1
        # for i in range(transformed.size(0)):  # Loop over the batch
        #     for j in range(
        #         transformed.size(1)
        #     ):  # Loop over the second dimension (4 in your case)
        #         transformed[i, j, top2_indices[i, j]] = 1

        # print(transformed)

        # print(transformed.shape)
        # print(top2_indices.shape)

        # one = torch.tensor(1.0).to(ut.get_device())
        # zero = torch.tensor(0.0).to(ut.get_device())
        # transformed = torch.where(probs > 0.3, one, zero)

        # print("decoded:")
        # print(decoded.shape)
        return transformed


class AutoregressiveLSTM(torch.nn.Module):
    def __init__(self, codebook_dim, hidden_dim, num_layers):
        super(AutoregressiveLSTM, self).__init__()
        self.name = "vqvae-lstm"
        self.codebook_dim = codebook_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = torch.nn.LSTM(
            codebook_dim, hidden_dim, num_layers, batch_first=True
        )

        # Output layer
        self.fc = torch.nn.Linear(hidden_dim, codebook_dim)

        # Loss function
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [batch, 4, hidden_dim]
        lstm_out = lstm_out[:, -1, :]  # [batch, hidden_dim]
        logits = self.fc(lstm_out)  # [batch, codebook_dim]
        return logits

    def calculate_loss(self, x, targets):
        # x: Input embeddings
        # targets: One-hot encoded target embeddings
        logits = self.forward(x)
        return self.loss_fn(logits, targets)

    def sample(self, initial_notes, sequence_length):
        """
        Generate a sequence of notes, starting from initial notes.

        :param initial_notes: The starting notes (one-hot encoded). Shape: [batch, init_len, codebook_dim]
        :param sequence_length: Total length of the sequence to generate.
        :return: Generated sequence of notes. [batch, sequence_length, codebook_dim]
        """
        self.eval()  # Set the model to evaluation mode

        # Initialize the sequence with the initial notes
        current_sequence = initial_notes

        # Loop to generate the rest of the sequence
        for _ in range(sequence_length - initial_notes.size(1)):
            # Get the logits for the next note
            logits = self.forward(current_sequence)
            # Convert logits to probabilities and sample the next note
            # You can use different strategies here, like argmax or sampling
            next_note_prob = torch.softmax(logits, dim=-1)
            next_note = torch.multinomial(next_note_prob, num_samples=1)
            one_hot_next_note = F.one_hot(next_note, num_classes=self.codebook_dim).to(
                next_note_prob.dtype
            )

            # Append the sampled note to the current sequence
            # print(one_hot_next_note.shape)
            current_sequence = torch.cat([current_sequence, one_hot_next_note], dim=1)

            # If your sequence is very long, you might want to limit the length of current_sequence
            # that you feed into the model to avoid excessive computational cost

        return current_sequence
