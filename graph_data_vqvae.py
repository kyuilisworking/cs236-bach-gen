import matplotlib.pyplot as plt
import numpy as np


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def plot_losses(losses, smooth_factor=5):
    for embedding, loss in losses.items():
        loss_smoothed = smooth(loss, smooth_factor)
        plt.plot(loss_smoothed, label=f"Embedding {embedding}")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("Reconstruction Loss for Different Embedding Spaces")
    plt.legend()
    plt.savefig("smoothed_reconstruction_losses_plot.png")
    plt.show()


# Now we modify the read_losses_and_plot function to include smoothing
def read_losses_and_plot(file_path, smooth_factor=5):
    losses = {}

    with open(file_path, "r") as file:
        for line in file:
            num_embeddings, epoch, recon_loss = line.strip().split(", ")
            num_embeddings = int(num_embeddings)
            recon_loss = float(recon_loss)

            if num_embeddings not in losses:
                losses[num_embeddings] = []
            losses[num_embeddings].append(recon_loss)

    plot_losses(losses, smooth_factor)


# Usage example with smoothing
file_path = "reconstruction_losses.txt"  # Replace with your actual file path
read_losses_and_plot(
    file_path, smooth_factor=5
)  # Increase the smoothing factor for more smoothing
