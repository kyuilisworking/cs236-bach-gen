import torch.optim as optim
from codebase.load_data import get_data_loader
from codebase.models.vqvae import VQVAE
import codebase.utils as ut

device = ut.get_device()

# Model parameters
config_file_path = "./config.json"  # Replace with your JSON file path
config = ut.read_model_config(config_file_path)

embedding_dim = config["embedding_dim"]
num_embeddings = config["num_embeddings"]
commitment_cost = config["commitment_cost"]
decay = config["decay"]

batch_size = 64


# Create the model
model = VQVAE(
    embedding_dim=embedding_dim,
    num_embeddings=num_embeddings,
    commitment_cost=commitment_cost,
    decay=decay,
)

model.to(device)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize the learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", factor=0.5, patience=3, verbose=True
)

# Create the DataLoader
dataloader = get_data_loader(
    "data/training_data/polyphonic_data_vqvae.pkl", batch_size=24, shuffle=True
)

# Training loop...
num_epochs = 1000
iter_save = 10
step = 0
for epoch in range(num_epochs):
    epoch_loss = 0.0

    for i, x in enumerate(dataloader):
        x = x.to(device).view(-1, 4, 88)  # .unsqueeze(1)  # [batch, 1, 4, 88]
        optimizer.zero_grad()

        # Forward pass
        total_loss, vq_loss, recon_loss = model.loss(x)
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

        step += 1

        if step % iter_save == 0:
            print("saving...")
            ut.save_model_by_name(model, step)

        if step % 1 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}]\nTotal loss: {total_loss}\n vq_loss: {vq_loss}\n recon_loss: {recon_loss}"
            )

    # Average loss for the epoch
    epoch_loss /= len(dataloader)
    print(f"Epoch loss: {epoch_loss}")

    # Update the learning rate scheduler
    scheduler.step(epoch_loss)

    # print(
    #     f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(dataloader)}], Summary: {summaries}"
    # )
    # ut.save_model_by_name(model, step)
