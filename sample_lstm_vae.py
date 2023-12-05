"""
Inference:
1) Load the model
2) Sample 200 z's from the prior
3) Use the z's to feed into the decoder to get logits
4) Use logits to calculate probabilities
5) Use probabilities to calculate x_gen

Visualization:

2) Visualize the list by putting it into a grid of 10x20
"""

import argparse
import numpy as np
import torch
import tqdm
from codebase import utils as ut
from codebase.models.lstm_vae import LstmVAE
from codebase.models.nns.params import EncoderConfig, DecoderConfig
from codebase.train import train
from pprint import pprint
from torchvision import datasets, transforms
from codebase.preprocess_data import reconstruct_midi_from_vectors
import torch.nn.functional as F

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--checkpoint", type=int, default=0, help="Number of training iterations"
)
parser.add_argument(
    "--count", type=int, default=200, help="Number of inferences to run"
)
args = parser.parse_args()

pprint(vars(args))
print("Model name:", "lstm_vae")

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

input_dim = 129
z_dim = 512
encoder_hidden_dim = 100

# Model parameters
conductor_hidden_dim = 100
conductor_layers = 1
embedding_dim = 100
decoder_hidden_dim = 100
decoder_layers = 1
num_subsequences = 4
subsequence_length = 8
teacher_force = True
teacher_force_prop = 1.0
kl_anneal_epochs = 200  # Number of epochs over which to linearly increase alpha
anneal_pct = 0.0  # Starting value for alpha
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
)

lstm_vae = LstmVAE(encoderConfig=encoder_config, decoderConfig=decoder_config).to(
    device
)

# load the model from the check point
ut.load_model_by_name(lstm_vae, global_step=args.checkpoint, device=device)

# sample z
z = lstm_vae.sample_z(10)
x = lstm_vae.sample_x_given(z, temperature=0.8)
one_hot_encoded = F.one_hot(x, num_classes=129)

print(x.shape)

print(x)

one_hot_encoded = one_hot_encoded.tolist()
print(x)

reconstruct_midi_from_vectors(one_hot_encoded, "./samples/sample.mid", 0.15)
