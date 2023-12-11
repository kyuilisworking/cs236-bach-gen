import torch.optim as optim
from codebase.load_data import get_data_loader
from codebase.models.lstm_vae import LstmVAE
from codebase.models.nns.params import EncoderConfig, DecoderConfig
import codebase.utils as ut
import matplotlib.pyplot as plt
import json

# ... [Your existing imports and configurations]


# def plot_losses(losses):
#     for embedding, loss in losses.items():
#         plt.plot(loss, label=f"Embedding {embedding}")
#     plt.xlabel("Epoch")
#     plt.ylabel("Reconstruction Loss")
#     plt.title("Reconstruction Loss for Different Num Embeddings")
#     plt.legend()

#     # Save the plot to a file
#     plt.savefig("reconstruction_losses_plot.png")
#     plt.show()


device = ut.get_device()

# Model parameters
# Model parameters
input_dim = 88
encoder_hidden_dim = 100

conductor_hidden_dim = 100
conductor_layers = 1
decoder_hidden_dim = 100
decoder_layers = 1
num_subsequences = 2
subsequence_length = 16
teacher_force = True
teacher_force_prop = 1.0
kl_anneal_epochs = 500  # Number of epochs over which to linearly increase alpha
anneal_pct = 0.2  # Starting value for alpha
anneal_pct_increment = 1.0 / kl_anneal_epochs  # Increment for each epoch


z_dims = [32, 64, 128, 256, 512]
all_losses = {}
recon_errors = {}
kl_divs = {}

batch_size = 32

# Create the DataLoader
dataloader = get_data_loader(
    "data/training_data/polyphonic_data_vqvae.pkl", batch_size=batch_size, shuffle=True
)

with open("hierarchical_nelbo.txt", "w") as nelbo_file, open(
    "hierarchical_recon.txt", "w"
) as recon_file, open("hierarchical_kl.txt", "w") as kl_file:
    for z_dim in z_dims:
        encoder_config = EncoderConfig(
            input_dim=input_dim, hidden_dim=encoder_hidden_dim, z_dim=z_dim
        )
        decoder_config = DecoderConfig(
            decoder_type="categorical",
            z_dim=z_dim,
            conductor_hidden_dim=conductor_hidden_dim,
            conductor_layers=conductor_layers,
            decoder_hidden_dim=decoder_hidden_dim,
            decoder_layers=decoder_layers,
            num_subsequences=num_subsequences,
            subsequence_length=subsequence_length,
            vocab_size=input_dim,
            teacher_force=teacher_force,
            teacher_force_prob=teacher_force_prop,
        )

        model = LstmVAE(encoderConfig=encoder_config, decoderConfig=decoder_config)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 30
        iter_save = 100
        step = 0
        losses = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_kl_div = 0.0
            epoch_recon_error = 0.0

            for i, x in enumerate(dataloader):
                # print(x.shape)
                x = (
                    x.to(device).view(batch_size, -1, 88).view(-1, 32, 88)
                )  # .unsqueeze(1)  # [batch, 1, 4, 88]
                optimizer.zero_grad()

                # Forward pass
                loss, summaries = model.loss(x)
                recon_error = summaries["gen/rec"]
                kl_z = summaries["gen/kl_z"]

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_kl_div += kl_z.item()
                epoch_recon_error += recon_error.item()

                step += 1

            # Average loss for the epoch
            epoch_loss /= len(dataloader)
            epoch_kl_div /= len(dataloader)
            epoch_recon_error /= len(dataloader)

            print(
                f"Epoch [{epoch+1}/{num_epochs}]\nNELBO loss: {epoch_loss}\nrecon error: {epoch_recon_error}\nkl_z: {epoch_kl_div}\n"
            )
            nelbo_file.write(f"{z_dim}, {epoch}, {epoch_loss}\n")
            recon_file.write(f"{z_dim}, {epoch}, {epoch_recon_error}\n")
            kl_file.write(f"{z_dim}, {epoch}, {epoch_kl_div}\n")

            # losses.append(recon_loss.item())

        # all_losses[num_embeddings] = losses


# plot_losses(all_losses)

# print(f"Epoch loss: {epoch_loss}")

# Update the learning rate scheduler
# scheduler.step(epoch_loss)

# print(
#     f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(dataloader)}], Summary: {summaries}"
# )
# ut.save_model_by_name(model, step)
