import torch
import torch.nn as nn
import torch.optim as optim
from codebase.load_data import get_data_loader
from codebase.models.lstm_vae import LstmVAE
from codebase.models.nns.params import EncoderConfig, DecoderConfig
import codebase.utils as ut

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Model parameters
input_dim = 129
z_dim = 512
encoder_hidden_dim = 1024
decoder_hidden_dim = 1024
num_subsequences = 2
subsequence_length = 16

encoder_config = EncoderConfig(
    input_dim=input_dim, hidden_dim=encoder_hidden_dim, z_dim=z_dim
)
decoder_config = DecoderConfig(
    z_dim=z_dim,
    conductor_hidden_dim=None,
    conductor_output_dim=None,
    vocab_size=input_dim,
    decoder_hidden_dim=decoder_hidden_dim,
    num_subsequences=num_subsequences,
    subsequence_length=subsequence_length,
)


# Create the model
model = LstmVAE(encoderConfig=encoder_config, decoderConfig=decoder_config)
model.to(device)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create the DataLoader
dataloader = get_data_loader(
    "data/training_data/32_note_sequences.pkl", batch_size=128, shuffle=True
)

# Training loop...
num_epochs = 1000
iter_save = 1000
for epoch in range(num_epochs):
    for i, x in enumerate(dataloader):
        x = x.to(device)
        optimizer.zero_grad()

        # Forward pass
        loss, summaries = model.loss(x)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
            )
        if i % iter_save == 0:
            ut.save_model_by_name(model, i)
