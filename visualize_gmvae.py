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
import torch.distributions as D
import tqdm
from codebase import utils as ut
from codebase.models.gmvae import GMVAE
from codebase.train import train
from pprint import pprint
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=10,    help="Number of latent dimensions")
parser.add_argument('--k',         type=int, default=500,   help="Number mixture components in MoG prior")
parser.add_argument('--checkpoint',  type=int, default=20000, help="Number of training iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--count', type=int, default=200, help="Number of inferences to run")
args = parser.parse_args()

layout = [
    ('model={:s}',  'gmvae'),
    ('z={:02d}',  args.z),
    ('k={:03d}',  args.k),
    ('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])

pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gmvae = GMVAE(z_dim=args.z, k=args.k, name=model_name).to(device)

# load the model from the check point
ut.load_model_by_name(gmvae, global_step=args.checkpoint, device=device)

# sample z from a mixture of Gaussians as a hierarchy of categorical and Gaussian
def sample_from_mixture(means, std_devs, weights=None, num_samples=1):
    """
    Sample from a mixture of Gaussians using PyTorch.
    
    Parameters:
    - means: tensor of means for each Gaussian component
    - std_devs: tensor of standard deviations for each Gaussian component
    - weights: tensor of weights for each Gaussian component, default is uniform
    - num_samples: number of samples to draw
    
    Returns:
    - samples: tensor of sampled values
    """
    n = means.shape[1]

    # If weights are not provided, set them to 1/n
    if weights is None:
      weights = torch.ones(n) / n
    
    # Step 1: Choose a Gaussian component based on the weights
    categorical_dist = D.Categorical(weights)
    chosen_gaussians = categorical_dist.sample((num_samples,))
    
    # Step 2: Sample from the chosen Gaussian
    samples = []
    for g in chosen_gaussians:
        normal_dist = D.Normal(means[:, g, :], std_devs[:, g, :])
        sample = normal_dist.sample()
        samples.append(sample)
    
    return torch.stack(samples)

m_mixture, v_mixture = ut.gaussian_parameters(gmvae.z_pre, dim=1)
z = sample_from_mixture(m_mixture, v_mixture, num_samples=args.count)

# infer x_gen
logits = gmvae.dec(z)

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