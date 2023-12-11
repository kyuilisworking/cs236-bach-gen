import matplotlib.pyplot as plt
import numpy as np


def plot_losses(losses, title, ylabel, out_path):
    for label, loss_data in losses.items():
        for z_dim, loss in loss_data.items():
            plt.plot(loss, label=f"{label} z_dim {z_dim}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(out_path)
    plt.show()


def read_losses(file_path, label):
    losses = {}
    with open(file_path, "r") as file:
        for line in file:
            num_embeddings, epoch, recon_loss = line.strip().split(", ")
            num_embeddings = int(num_embeddings)
            recon_loss = float(recon_loss)

            if num_embeddings not in losses:
                losses[num_embeddings] = []
            losses[num_embeddings].append(recon_loss)
    return losses


def read_losses_and_plot(file_path_1, file_path_2, title, ylabel, out_path):
    losses = {}
    losses["One Layer"] = read_losses(file_path_1, "One Layer")
    losses["Hierarchical"] = read_losses(file_path_2, "Hierarchical")

    plot_losses(losses, title, ylabel, out_path)


# Usage example
# file_path_1 = "one_layer_recon.txt"  # Replace with your actual file path
# file_path_2 = "hierarchical_recon.txt"  # Replace with your actual file path
# title = "Reconstruction Errors of LSTM VAEs"
# ylabel = "Reconstruction Error"
# out_path = "lstm_vaecombined_losses_plot.png"

file_path_1 = "one_layer_kl.txt"  # Replace with your actual file path
file_path_2 = "hierarchical_kl.txt"  # Replace with your actual file path
title = "KL Divergences of the prior p(z) of LSTM VAEs"
ylabel = "KL(q(z|x) | p(z))"
out_path = "lstm_vae_combined_kl_z_plot.png"

read_losses_and_plot(file_path_1, file_path_2, title, ylabel, out_path)
