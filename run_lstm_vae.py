import torch
import torch.nn as nn
import torch.optim as optim
from codebase.load_data import get_data_loader
from codebase.models.lstm_vae import LstmVAE
from codebase.models.nns.params import EncoderConfig, DecoderConfig
import codebase.utils as ut

device = ut.get_device()

# Model parameters
input_dim = 129
z_dim = 512
encoder_hidden_dim = 100

conductor_hidden_dim = 100
conductor_layers = 1
embedding_dim = 100
decoder_hidden_dim = 100
decoder_layers = 1
num_subsequences = 4
subsequence_length = 8
teacher_force = True
teacher_force_prop = 1.0
kl_anneal_epochs = 500  # Number of epochs over which to linearly increase alpha
anneal_pct = 0.2  # Starting value for alpha
anneal_pct_increment = 1.0 / kl_anneal_epochs  # Increment for each epoch


encoder_config = EncoderConfig(
    input_dim=input_dim, hidden_dim=encoder_hidden_dim, z_dim=z_dim
)
decoder_config = DecoderConfig(
    decoder_type="hierarchical",
    z_dim=z_dim,
    conductor_hidden_dim=conductor_hidden_dim,
    conductor_layers=conductor_layers,
    embedding_dim=embedding_dim,
    decoder_hidden_dim=decoder_hidden_dim,
    decoder_layers=decoder_layers,
    num_subsequences=num_subsequences,
    subsequence_length=subsequence_length,
    vocab_size=input_dim,
    teacher_force=teacher_force,
    teacher_force_prob=teacher_force_prop,
)


# Create the model
model = LstmVAE(encoderConfig=encoder_config, decoderConfig=decoder_config)
model.to(device)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create the DataLoader
dataloader = get_data_loader(
    "data/training_data/32_note_sequences_filtered.pkl", batch_size=256, shuffle=True
)

# Training loop...
num_epochs = 1000
iter_save = 1000
step = 0
for epoch in range(num_epochs):
    print(anneal_pct)

    for i, x in enumerate(dataloader):
        x = x.to(device)
        optimizer.zero_grad()

        # Forward pass
        loss, summaries = model.loss(x, anneal_pct=anneal_pct)
        loss.backward()
        optimizer.step()

        step += 1

        # if step % iter_save == 0:
        #     print("saving...")
        #     ut.save_model_by_name(model, step)

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(dataloader)}], Summary: {summaries}"
    )
    ut.save_model_by_name(model, step)

    anneal_pct = min(1.0, anneal_pct + anneal_pct_increment)
