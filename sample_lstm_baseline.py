import argparse
import numpy as np
import torch
from codebase import utils as ut
from codebase.models.lstm_baseline import LstmBaseline
from codebase.train import train
from pprint import pprint
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
print("Model name:", "lstm_baseline")

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Model parameters
input_dim = 129
hidden_dim = 512
num_layers = 2

# Create the model
model = LstmBaseline(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
model.to(device)

# load the model from the check point
ut.load_model_by_name(model, global_step=args.checkpoint, device=device)


# Generate a random sequence.
def create_initial_note_seq(input_dim, note_idx):
    initial_note = torch.zeros(1, 1, input_dim)

    # Set the kth value to 1
    initial_note[0, 0, note_idx] = 1

    return initial_note


initial_notes = create_initial_note_seq(input_dim, 56).to(device)
generated_seq = model.sample(initial_notes, 32)
generated_seq = generated_seq.tolist()

reconstruct_midi_from_vectors(generated_seq, "./samples/baseline_sample.mid", 0.20)
