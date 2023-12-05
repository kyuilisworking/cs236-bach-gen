import argparse
import torch
from codebase import utils as ut
from pprint import pprint
from codebase.models.vqvae import VQVAE
from codebase.load_data import get_data_loader
import os


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--checkpoint", type=int, default=0, help="Number of training iterations"
)

args = parser.parse_args()

pprint(vars(args))
print("Model name:", "vqvae")

device = ut.get_device()

# Model parameters
config_file_path = "config.json"  # Replace with your JSON file path
config = ut.read_model_config(config_file_path)

embedding_dim = config["embedding_dim"]
num_embeddings = config["num_embeddings"]
commitment_cost = config["commitment_cost"]
decay = config["decay"]


# Create the model
model = VQVAE(
    embedding_dim=embedding_dim,
    num_embeddings=num_embeddings,
    commitment_cost=commitment_cost,
    decay=decay,
)

model.to(device)

# Create the DataLoader
dataloader = get_data_loader(
    "data/training_data/polyphonic_data_vqvae.pkl", batch_size=16, shuffle=True
)

# load the model from the check point
ut.load_model_by_name(model, global_step=args.checkpoint, device=device)

# For each batch, [batch, 64, 4, 88], generate a sequence of embeddings of
# shape [batch, 64, embedding_dim], and then generate prediction data for the
# autoregressive model

input_sequences = []
targets = []
save_interval = 10  # Number of batches after which to save

for i, x in enumerate(dataloader):
    with torch.no_grad():
        model.eval()
        x = x.view(-1, 4, 88).to(device)  # .unsqueeze(1)
        embeddings_one_hot = model.generate_embeddings_one_hot(
            x
        )  # [batch, num_embeddings]
        embeddings_one_hot = embeddings_one_hot.view(-1, 64, num_embeddings)

        # now we have a sequence of 64 embeddings in one-hot format
        w = 4  # window size

        # create a dataset for the autoregressive model in the following way:
        """
        1) Do a sliding window to create a training set of 4 elements in the sequence followed by a target next embedding
        2) Thereby generate 64 - w - 1 examples for training.
        3) Figure out an appropriate way to save this data
        """

        for j in range(64 - w - 1):
            input_seq = embeddings_one_hot[
                :, j : j + w, :
            ]  # [batch, w, num_embeddings]
            target_seq = embeddings_one_hot[
                :, j + w, :
            ]  # [batch, num_embeddings] (next step)
            input_sequences.append(input_seq)
            targets.append(target_seq)

input_sequences = torch.cat(input_sequences, dim=0)
targets = torch.cat(targets, dim=0)
ut.save_data_with_pickle(
    input_sequences,
    targets,
    f"./data/training_data/vqvae_lstm_train_{args.checkpoint}.pkl",
)
