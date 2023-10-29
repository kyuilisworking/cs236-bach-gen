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
from codebase.models.vae import VAE
from codebase.train import train
from pprint import pprint
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=10,    help="Number of latent dimensions")
parser.add_argument('--checkpoint',  type=int, default=20000, help="Number of training iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--count', type=int, default=200, help="Number of inferences to run")
args = parser.parse_args()

layout = [
    ('model={:s}',  'vae'),
    ('z={:02d}',  args.z),
    ('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])

pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = VAE(z_dim=args.z, name=model_name).to(device)

# load the model from the check point
ut.load_model_by_name(vae, global_step=args.checkpoint, device=device)

# sample z
expanded_mean = vae.z_prior_m.data.expand(args.count, args.z)
z_prior_std = torch.sqrt(vae.z_prior_v)
expanded_std = z_prior_std.expand(args.count, args.z)
z = torch.normal(mean=expanded_mean, std=expanded_std).to(device)

# infer x_gen
logits = vae.dec(z)

# probabilities
probs = torch.sigmoid(logits)

# generated x
x_gen = (probs > 0.5)

images = x_gen.reshape(args.count, 28, 28)

def plot_images_grid(images, save_path=None):
    plt.figure(figsize=(20, 10))

    for idx, image in enumerate(images):
        plt.subplot(10, 20, idx+1)
        plt.imshow(image, cmap='gray') 
        plt.axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()

plot_images_grid(images, './images/'+model_name)