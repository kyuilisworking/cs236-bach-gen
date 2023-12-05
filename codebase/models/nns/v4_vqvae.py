import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5
    ):
        super(VectorQuantizerEMA, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.register_buffer(
            "embeddings", torch.randn(embedding_dim, num_embeddings) * 2 - 1
        )
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_dw", torch.zeros(embedding_dim, num_embeddings))

    def forward(self, inputs):
        # Flatten input
        flat_inputs = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_inputs**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings**2, dim=0)
            - 2 * torch.matmul(flat_inputs, self.embeddings)
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(
            flat_inputs.dtype
        )
        encoding_indices = encoding_indices.view(*inputs.shape[:-1])
        quantized = torch.matmul(encodings, self.embeddings.t()).view_as(inputs)

        # Use EMA to update the embedding vectors
        if self.training:
            self.ema_cluster_size = self.decay * self.ema_cluster_size + (
                1 - self.decay
            ) * torch.sum(encodings, 0)
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )

            dw = torch.matmul(flat_inputs.t(), encodings)
            self.ema_dw = self.decay * self.ema_dw + (1 - self.decay) * dw

            self.embeddings = self.ema_dw / self.ema_cluster_size.unsqueeze(0)
            # normalize embeddings
            self.embeddings = F.normalize(self.embeddings, p=2, dim=0)
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        return loss, quantized, encoding_indices

    def encode_one_hot(self, inputs):
        """
        Convert input embeddings to one-hot encodings based on the nearest embeddings in the codebook.
        :param inputs: A tensor of embeddings with shape [batch_size, embedding_dim].
        :return: A tensor of one-hot encoded vectors with shape [batch_size, num_embeddings].
        """
        # Flatten input
        flat_inputs = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_inputs**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings**2, dim=0)
            - 2 * torch.matmul(flat_inputs, self.embeddings)
        )

        # Find the nearest embeddings
        encoding_indices = torch.argmin(distances, dim=1)

        # Convert to one-hot encodings
        one_hot_encodings = F.one_hot(encoding_indices, self.num_embeddings).type(
            flat_inputs.dtype
        )

        return one_hot_encodings

    def decode_one_hot(self, one_hot):
        """
        Convert input one-hot representations of embeddings to actual embedding vectors.
        :param one_hot: A tensor of one-hot embeddings with shape [batch_size, num_embeddings].
        :return: A tensor of embedding vectors with shape [batch_size, embedding_dim].
        """
        # Convert one-hot encodings to embedding vectors
        embedding_vectors = torch.matmul(one_hot, self.embeddings.t())

        return embedding_vectors


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        self.initial_conv = nn.Conv2d(1, 256, kernel_size=(4, 4), stride=2, padding=1)
        self.residual_block1 = ResidualBlock(256)
        self.residual_block2 = ResidualBlock(256)
        self.final_conv = nn.Conv2d(
            256, embedding_dim, kernel_size=(2, 44)
        )  # Adjusted to fit the input size

    def forward(self, x):
        device = next(self.initial_conv.parameters()).device
        x = self.initial_conv(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.final_conv(x)
        return x.view(x.size(0), -1)  # Flatten the output


class Decoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Decoder, self).__init__()
        # Reverse the final convolution of the encoder
        self.initial_deconv = nn.ConvTranspose2d(
            embedding_dim, 256, kernel_size=(2, 44)
        )

        # Two residual blocks (same as in the encoder)
        self.residual_block1 = ResidualBlock(256)
        self.residual_block2 = ResidualBlock(256)

        # Reverse the initial convolution of the encoder
        self.final_deconv = nn.ConvTranspose2d(
            256, 1, kernel_size=(4, 4), stride=2, padding=1
        )

    def forward(self, x):
        # Reshape from flat latent representation to 2D
        x = x.view(
            x.size(0), -1, 1, 1
        )  # Adjust shape as needed to fit the initial_deconv layer
        x = self.initial_deconv(x)

        x = self.residual_block1(x)
        x = self.residual_block2(x)

        x = self.final_deconv(x)
        return x.squeeze(1)  # Remove channel dimension, resulting in [batch, 4, 88]


class EnhancedEncoder(nn.Module):
    def __init__(self, embedding_dim=128):  # embedding_dim set to 128
        super(EnhancedEncoder, self).__init__()

        # Define Convolutional Layers
        # Keeping the previous layers as they are
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(2, 4), stride=(2, 4), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(2, 4), stride=(1, 2), padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(2, 2), stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(128)

        # # Define Residual Blocks
        # self.residual_block1 = ResidualBlock(256)
        # self.residual_block2 = ResidualBlock(256)

        # Final Convolutional Layer to reduce the number of channels to embedding_dim
        # Adjust the kernel size based on the output size of conv3 and residual blocks
        self.final_conv = nn.Conv2d(128, embedding_dim, kernel_size=(1, 9))
        self.bn4 = nn.BatchNorm2d(embedding_dim)

    def forward(self, x):
        # Initial shape of x: [batch, 1, 4, 88]

        x = self.bn1(F.relu(self.conv1(x)))
        # Shape after conv1: [batch, 128, 3, 44]
        # print(f"1: {x.shape}")
        x = self.bn2(F.relu(self.conv2(x)))
        # Shape after conv2: [batch, 256, 2, 22]
        # print(f"2: {x.shape}")

        x = self.bn3(F.relu(self.conv3(x)))
        # Shape after conv3: [batch, 512, 1, 11]
        # print(f"3: {x.shape}")

        # x = self.residual_block1(x)
        # x = self.residual_block2(x)

        # print(f"res: {x.shape}")

        x = self.bn4(F.relu(self.final_conv(x)))
        # print(f"4: {x.shape}")
        # Shape after final_conv: [batch, embedding_dim, 1, 1]

        x = x.view(x.size(0), -1)
        # Final shape: [batch, embedding_dim]

        return x


class EnhancedDecoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(EnhancedDecoder, self).__init__()

        # Initial layer to start the upsampling process
        self.initial_layer = nn.ConvTranspose2d(embedding_dim, 128, kernel_size=(1, 9))
        self.bn0 = nn.BatchNorm2d(128)

        # Residual blocks
        # self.residual_block1 = ResidualBlock(256)
        # self.residual_block2 = ResidualBlock(256)

        # Adjust these layers to correctly upsample
        self.conv_transpose1 = nn.ConvTranspose2d(
            128, 64, kernel_size=(2, 2), stride=1, padding=0
        )
        self.bn1 = nn.BatchNorm2d(64)
        # The output padding is often needed to fine-tune the output dimensions
        self.conv_transpose2 = nn.ConvTranspose2d(
            64, 32, kernel_size=(2, 4), stride=(1, 2), padding=0
        )
        self.bn2 = nn.BatchNorm2d(32)
        self.conv_transpose3 = nn.ConvTranspose2d(
            32, 1, kernel_size=(2, 4), stride=(2, 4), padding=(1, 0)
        )
        self.bn3 = nn.BatchNorm2d(1)

    def forward(self, x):
        # Initial reshape
        x = x.view(x.size(0), -1, 1, 1)

        x = self.bn0(F.relu(self.initial_layer(x)))

        # x = self.residual_block1(x)
        # x = self.residual_block2(x)

        x = self.bn1(F.relu(self.conv_transpose1(x)))
        x = self.bn2(F.relu(self.conv_transpose2(x)))
        x = self.bn3(F.relu(self.conv_transpose3(x)))

        return x.squeeze(1)


class Encoder1D(nn.Module):
    def __init__(self, embedding_dim):
        super(Encoder1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm1d(32)
        self.fc = nn.Linear(32 * 88, embedding_dim)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the output for the linear layer
        x = self.fc(x)
        return x


class Decoder1D(nn.Module):
    def __init__(self, embedding_dim):
        super(Decoder1D, self).__init__()
        self.fc = nn.Linear(embedding_dim, 32 * 88)
        self.conv1 = nn.Conv1d(
            in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm1d(4)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 32, 88)  # Reshape to fit for convolution layers
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(torch.sigmoid(self.conv2(x)))  # Sigmoid for multihot output
        return x
