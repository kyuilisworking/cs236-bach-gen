import torch
from torch.utils.data import Dataset, DataLoader
import pickle


class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        x = torch.tensor(sequence, dtype=torch.float32)  # Only Input sequence
        return x


def get_data_loader(file_path, batch_size=32, shuffle=True):
    with open(file_path, "rb") as file:
        sequences = pickle.load(file)

    dataset = SequenceDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
