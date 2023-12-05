import argparse
import torch
import torch.nn
from codebase import utils as ut
from pprint import pprint
from codebase.models.vqvae import VQVAE, AutoregressiveLSTM
from codebase.load_data import get_data_loader
from codebase.preprocess_data_vqvae import reconstruct_midi_from_multihot_seq
import torch.nn.functional as F

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--vqvae_checkpoint", type=int, default=80, help="Number of training iterations"
)
parser.add_argument(
    "--lstm_checkpoint", type=int, default=10, help="Number of training iterations"
)

args = parser.parse_args()

pprint(vars(args))

device = ut.get_device()

# Model VQVAE parameters
config_file_path = "./config.json"  # Replace with your JSON file path
config = ut.read_model_config(config_file_path)

embedding_dim = config["embedding_dim"]
num_embeddings = config["num_embeddings"]
commitment_cost = config["commitment_cost"]
decay = config["decay"]

# LSTM parameters
hidden_dim = config["lstm_hidden_dim"]
num_layers = config["lstm_num_layers"]

vqvae = VQVAE(
    embedding_dim=embedding_dim,
    num_embeddings=num_embeddings,
    commitment_cost=commitment_cost,
    decay=decay,
)

lstm = AutoregressiveLSTM(num_embeddings, hidden_dim, num_layers)

vqvae.to(device)
lstm.to(device)

# Load from checkpoints
ut.load_model_by_name(vqvae, global_step=args.vqvae_checkpoint, device=device)
ut.load_model_by_name(lstm, global_step=args.lstm_checkpoint, device=device)

# Sample a batch of one-hot embeddings from vqvae
sample_size = 10
init_one_hot = vqvae.random_sample_one_hot_embedding(sample_size).unsqueeze(
    1
)  # [batch, 1, num_embeddings]
# print(init_one_hot.shape)

# Use LSTM to generate sequences from the initial one-hot embeddings
seq_len = 16
generated_embedding_seq = lstm.sample(
    initial_notes=init_one_hot, sequence_length=seq_len
)  # [batch, seq_len, num_embeddings]
# print(generated_embedding_seq)
indices = torch.argmax(generated_embedding_seq, dim=-1)
# print(indices)

# Send the generated embedding sequence through the decoder
generated_embedding_seq = generated_embedding_seq.reshape(-1, num_embeddings)
notes = vqvae.embedding_to_notes(generated_embedding_seq)  # [batch, 4, 88]

# print(notes.shape)

notes = notes.view(-1, seq_len, 4, 88)  # [batch, seq_len, 4, 88]
notes = notes.reshape(notes.shape[0], seq_len * 4, 88)

# print(notes.shape)
indices = torch.argmax(notes, dim=-1)
# print(indices)

for idx, seq in enumerate(notes):
    reconstruct_midi_from_multihot_seq(
        seq, output_midi_path=f"./samples/vqvae/{idx}.mid", sixteenth_note_duration=0.2
    )

all_one_hot_embeddings = []
for i in range(num_embeddings):
    one_hot = F.one_hot(torch.tensor([i]), num_classes=num_embeddings)
    all_one_hot_embeddings.append(one_hot)

all_one_hot_embeddings = torch.cat(all_one_hot_embeddings, dim=0).to(device).float()
# print(f"all_one_hot shape: {all_one_hot_embeddings.shape}")

all_embedding_maps = vqvae.embedding_to_notes(all_one_hot_embeddings)

for i, notes in enumerate(all_embedding_maps):
    note_indices = []
    for note in notes:
        # print(note)
        indices = torch.nonzero(note, as_tuple=True)[0]
        note_indices.append(indices)
    # Convert indices to a list and print it
    print(f"Embedding {i}")
    print(note_indices)
    # print("\n")
