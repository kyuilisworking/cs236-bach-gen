# Copyright (c) 2021 Rui Shu
import numpy as np
import os
import shutil
import sys
import torch
import pickle
import json

# import tensorflow as tf
from codebase.models.vae import VAE
from torch.nn import functional as F
from torchvision import datasets, transforms

bce = torch.nn.BCEWithLogitsLoss(reduction="none")

################################################################################
# Please familiarize yourself with the code below.
#
# Note that the notation is
# argument: argument_type: argument_shape
#
# Furthermore, the expected argument_shape is only a guideline. You're free to
# pass in inputs that violate the expected argument_shape provided you know
# what you're doing
################################################################################


def sample_gaussian(m, v):
    """
    Element-wise application reparameterization trick to sample from Gaussian

    Args:
        m: tensor: (batch, ...): Mean
        v: tensor: (batch, ...): Variance

    Return:
        z: tensor: (batch, dim): Samples
    """
    ################################################################################
    # TODO: Modify/complete the code here
    # Sample z
    ################################################################################
    # Sample from a standard Gaussian with the same shape as m
    # elem-wise multiplication with v
    # elem-wise addition with m and return
    z = torch.randn_like(m) * torch.sqrt(v) + m
    ################################################################################
    # End of code modification
    ################################################################################
    return z


def log_normal(x, m, v):
    """
    Computes the elem-wise log probability of a Gaussian and then sum over the
    last dim. Basically we're assuming all dims are batch dims except for the
    last dim.

    Args:
        x: tensor: (batch_1, batch_2, ..., batch_k, dim): Observation
        m: tensor: (batch_1, batch_2, ..., batch_k, dim): Mean
        v: tensor: (batch_1, batch_2, ..., batch_k, dim): Variance

    Return:
        log_prob: tensor: (batch_1, batch_2, ..., batch_k): log probability of
            each sample. Note that the summation dimension is not kept
    """
    ################################################################################
    # TODO: Modify/complete the code here
    # Compute element-wise log probability of normal and remember to sum over
    # the last dimension
    ################################################################################
    d = x.shape[-1]
    two_pi = torch.tensor(2 * np.pi)
    log_prob = (
        -d / 2 * torch.log(two_pi)
        - 0.5 * torch.sum(torch.log(v), dim=-1)
        + torch.sum(-((x - m) ** 2) / (2 * v), dim=-1)
    )
    ################################################################################
    # End of code modification
    ################################################################################
    return log_prob


def log_normal_mixture(z, m, v):
    """
    Computes log probability of Gaussian mixture.

    Args:
        z: tensor: (batch, dim): Observations
        m: tensor: (batch, mix, dim): Mixture means
        v: tensor: (batch, mix, dim): Mixture variances

    Return:
        log_prob: tensor: (batch,): log probability of each sample
    """
    ################################################################################
    # TODO: Modify/complete the code here
    # Compute the uniformly-weighted mixture of Gaussians density for each sample
    # in the batch
    ################################################################################
    d = z.shape[-1]
    two_pi = torch.tensor(2 * np.pi)
    log_prob = -d / 2 * torch.log(two_pi) + log_mean_exp(
        torch.sum(
            -torch.log(torch.sqrt(v)) - (z.unsqueeze(1) - m) ** 2 / (2 * v), dim=-1
        ),
        dim=-1,
    )
    ################################################################################
    # End of code modification
    ################################################################################
    return log_prob


def gaussian_parameters(h, dim=-1):
    """
    Converts generic real-valued representations into mean and variance
    parameters of a Gaussian distribution

    Args:
        h: tensor: (batch, ..., dim, ...): Arbitrary tensor
        dim: int: (): Dimension along which to split the tensor for mean and
            variance

    Returns:
        m: tensor: (batch, ..., dim / 2, ...): Mean
        v: tensor: (batch, ..., dim / 2, ...): Variance
    """
    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v


def log_bernoulli_with_logits(x, logits):
    """
    Computes the log probability of a Bernoulli given its logits

    Args:
        x: tensor: (batch, dim): Observation
        logits: tensor: (batch, dim): Bernoulli logits

    Return:
        log_prob: tensor: (batch,): log probability of each sample
    """
    log_prob = -bce(input=logits, target=x).sum(-1)
    return log_prob


def kl_cat(q, log_q, log_p):
    """
    Computes the KL divergence between two categorical distributions

    Args:
        q: tensor: (batch, dim): Categorical distribution parameters
        log_q: tensor: (batch, dim): Log of q
        log_p: tensor: (batch, dim): Log of p

    Return:
        kl: tensor: (batch,) kl between each sample
    """
    element_wise = q * (log_q - log_p)
    kl = element_wise.sum(-1)
    return kl


def kl_normal(qm, qv, pm, pv):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension

    Args:
        qm: tensor: (batch, dim): q mean
        qv: tensor: (batch, dim): q variance
        pm: tensor: (batch, dim): p mean
        pv: tensor: (batch, dim): p variance

    Return:
        kl: tensor: (batch,): kl between each sample
    """
    element_wise = 0.5 * (
        torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1
    )
    kl = element_wise.sum(-1)
    return kl


def duplicate(x, rep):
    """
    Duplicates x along dim=0

    Args:
        x: tensor: (batch, ...): Arbitrary tensor
        rep: int: (): Number of replicates. Setting rep=1 returns orignal x

    Returns:
        _: tensor: (batch * rep, ...): Arbitrary replicated tensor
    """
    return x.expand(rep, *x.shape).reshape(-1, *x.shape[1:])


def log_mean_exp(x, dim):
    """
    Compute the log(mean(exp(x), dim)) in a numerically stable manner

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which mean is computed

    Return:
        _: tensor: (...): log(mean(exp(x), dim))
    """
    return log_sum_exp(x, dim) - np.log(x.size(dim))


def log_sum_exp(x, dim=0):
    """
    Compute the log(sum(exp(x), dim)) in a numerically stable manner

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which sum is computed

    Return:
        _: tensor: (...): log(sum(exp(x), dim))
    """
    max_x = torch.max(x, dim)[0]
    new_x = x - max_x.unsqueeze(dim).expand_as(x)
    return max_x + (new_x.exp().sum(dim)).log()


def load_model_by_name(model, global_step, device=None):
    """
    Load a model based on its name model.name and the checkpoint iteration step

    Args:
        model: Model: (): A model
        global_step: int: (): Checkpoint iteration
    """
    file_path = os.path.join(
        "checkpoints", model.name, "model-{:05d}.pt".format(global_step)
    )
    state = torch.load(file_path, map_location=device)
    model.load_state_dict(state)
    print("Loaded from {}".format(file_path))


################################################################################
# No need to read/understand code beyond this point. Unless you want to.
# But do you tho ¯\_(ツ)_/¯
################################################################################


def save_model_by_name(model, global_step):
    save_dir = os.path.join("checkpoints", model.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, "model-{:05d}.pt".format(global_step))
    state = model.state_dict()
    torch.save(state, file_path)
    print("Saved to {}".format(file_path))


def prepare_writer(model_name, overwrite_existing=False):
    log_dir = os.path.join("logs", model_name)
    save_dir = os.path.join("checkpoints", model_name)
    maybe_delete_existing(log_dir, overwrite_existing)
    maybe_delete_existing(save_dir, overwrite_existing)
    # Sadly, I've been told *not* to use tensorflow :<
    # writer = tf.summary.FileWriter(log_dir)
    writer = None
    return writer


def log_summaries(writer, summaries, global_step):
    pass  # Sad :<
    # for tag in summaries:
    #     val = summaries[tag]
    #     tf_summary = tf.Summary.Value(tag=tag, simple_value=val)
    #     writer.add_summary(tf.Summary(value=[tf_summary]), global_step)
    # writer.flush()


def maybe_delete_existing(path, overwrite_existing):
    if not os.path.exists(path):
        return

    if overwrite_existing:
        print("Deleting existing path: {}".format(path))
        shutil.rmtree(path)
    else:
        raise FileExistsError(
            """
    Unpermitted attempt to delete {}.
    1. To overwrite checkpoints and logs when re-running a model, remember to pass --overwrite 1 as argument.
    2. To run a replicate model, pass --run NEW_ID where NEW_ID is incremented from 0.""".format(
                path
            )
        )


def reset_weights(m):
    try:
        m.reset_parameters()
    except AttributeError:
        pass


def get_mnist_data(device, use_test_subset=True):
    preprocess = transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data", train=True, download=True, transform=preprocess),
        batch_size=97,  # Using a weird batch size to prevent students from hard-coding
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data", train=False, download=True, transform=preprocess),
        batch_size=97,
        shuffle=True,
    )

    # Create pre-processed training and test sets
    X_train = train_loader.dataset.train_data.to(device).reshape(-1, 784).float() / 255
    y_train = train_loader.dataset.train_labels.to(device)
    X_test = test_loader.dataset.test_data.to(device).reshape(-1, 784).float() / 255
    y_test = test_loader.dataset.test_labels.to(device)

    # Create supervised subset (deterministically chosen)
    # This subset will serve dual purpose of log-likelihood evaluation and
    # semi-supervised learning. Pretty hacky. Don't judge :<
    X = X_test if use_test_subset else X_train
    y = y_test if use_test_subset else y_train

    xl, yl = [], []
    for i in range(10):
        idx = y == i
        idx_choice = get_mnist_index(i, test=use_test_subset)
        xl += [X[idx][idx_choice]]
        yl += [y[idx][idx_choice]]

    if use_test_subset:
        xl = static_binarize(torch.cat(xl)).to(device)
    else:
        xl = torch.cat(xl).to(device)

    yl = torch.cat(yl).to(device)
    yl = yl.new(np.eye(10)[yl.cpu()]).to(device)
    labeled_subset = (xl, yl)
    return train_loader, labeled_subset, (X_test, y_test)


def static_binarize(x):
    # torch.bernoulli seeding behavior is different on CPU v GPU
    # so we'll convert to numpy array and use binomial to sample static x
    with FixedSeed(0):
        x = np.random.binomial(1, x.cpu().numpy())
        x = torch.FloatTensor(x)
    return x


def get_mnist_index(i, test=True):
    # Obviously *hand*-coded
    train_idx = np.array(
        [
            [2732, 2607, 1653, 3264, 4931, 4859, 5827, 1033, 4373, 5874],
            [5924, 3468, 6458, 705, 2599, 2135, 2222, 2897, 1701, 537],
            [2893, 2163, 5072, 4851, 2046, 1871, 2496, 99, 2008, 755],
            [797, 659, 3219, 423, 3337, 2745, 4735, 544, 714, 2292],
            [151, 2723, 3531, 2930, 1207, 802, 2176, 2176, 1956, 3622],
            [3560, 756, 4369, 4484, 1641, 3114, 4984, 4353, 4071, 4009],
            [2105, 3942, 3191, 430, 4187, 2446, 2659, 1589, 2956, 2681],
            [4180, 2251, 4420, 4870, 1071, 4735, 6132, 5251, 5068, 1204],
            [3918, 1167, 1684, 3299, 2767, 2957, 4469, 560, 5425, 1605],
            [5795, 1472, 3678, 256, 3762, 5412, 1954, 816, 2435, 1634],
        ]
    )

    test_idx = np.array(
        [
            [684, 559, 629, 192, 835, 763, 707, 359, 9, 723],
            [277, 599, 1094, 600, 314, 705, 551, 87, 174, 849],
            [537, 845, 72, 777, 115, 976, 755, 448, 850, 99],
            [984, 177, 755, 797, 659, 147, 910, 423, 288, 961],
            [265, 697, 639, 544, 543, 714, 244, 151, 675, 510],
            [459, 882, 183, 28, 802, 128, 128, 53, 550, 488],
            [756, 273, 335, 388, 617, 42, 442, 543, 888, 257],
            [57, 291, 779, 430, 91, 398, 611, 908, 633, 84],
            [203, 324, 774, 964, 47, 639, 131, 972, 868, 180],
            [1000, 846, 143, 660, 227, 954, 791, 719, 909, 373],
        ]
    )

    if test:
        return test_idx[i]

    else:
        return train_idx[i]


def get_svhn_data(device):
    preprocess = transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN("data", split="extra", download=True, transform=preprocess),
        batch_size=100,
        shuffle=True,
    )

    return train_loader, (None, None), (None, None)


def gumbel_softmax(logits, tau, eps=1e-8):
    U = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(U + eps) + eps)
    y = logits + gumbel
    y = F.softmax(y / tau, dim=1)
    return y


class FixedSeed:
    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        np.random.set_state(self.state)


def get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


# System verifier
if sys.version_info[0] < 3:
    raise Exception(
        "Detected unpermitted Python version: Python{}. You should use Python3.".format(
            sys.version_info[0]
        )
    )


def save_data_with_pickle(inputs, targets, save_path):
    """
    Appends the inputs and targets to a file using pickle.
    """
    data_to_save = {"inputs": inputs, "targets": targets}

    # Check if file exists. If not, create it.
    if not os.path.isfile(save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data_to_save, f)


def read_model_config(file_path):
    """
    Read model configuration from a JSON file.

    :param file_path: Path to the JSON file containing the model parameters.
    :return: A dictionary with model parameters.
    """
    with open(file_path, "r") as file:
        config = json.load(file)

    return config
